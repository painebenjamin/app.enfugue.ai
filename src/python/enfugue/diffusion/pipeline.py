from __future__ import annotations

from typing import (
    Any,
    List,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Iterator,
    Literal,
    Mapping,
    cast,
)

import os
import PIL
import PIL.Image
import copy
import math
import torch
import inspect
import datetime
import numpy as np
import safetensors.torch

from contextlib import contextmanager
from collections import defaultdict

from transformers import (
    AutoFeatureExtractor,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers.schedulers import (
    KarrasDiffusionSchedulers,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    create_unet_diffusers_config,
    create_vae_diffusers_config,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
)

from diffusers.utils import randn_tensor, PIL_INTERPOLATION
from diffusers.image_processor import VaeImageProcessor

from enfugue.util import logger, check_download_to_dir, TokenMerger

# This is ~64k×64k. Absurd, but I don't judge
PIL.Image.MAX_IMAGE_PIXELS = 2**32


class EnfugueStableDiffusionPipeline(StableDiffusionPipeline):
    """
    This pipeline merges all of the following:
    1. txt2img
    2. img2img
    3. inpainting/outpainting
    4. controlnet
    5. tensorrt
    """

    controlnet: Optional[ControlNetModel]
    text_encoder: Optional[CLIPTextModel]
    text_encoder_2: Optional[CLIPTextModelWithProjection]
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Optional[CLIPTextModel],
        text_encoder_2: Optional[CLIPTextModelWithProjection],
        tokenizer: Optional[CLIPTokenizer],
        tokenizer_2: Optional[CLIPTokenizer],
        unet: UNet2DConditionModel,
        controlnet: Optional[ControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetic_score: bool = False,
        force_full_precision_vae: bool = False,
        engine_size: int = 512,
        chunking_size: int = 64,
        chunking_blur: int = 64,
    ) -> None:
        super(EnfugueStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        # Save scheduler config for hotswapping
        self.scheduler_config = {**dict(scheduler.config)}

        # Enfugue engine settings
        self.engine_size = engine_size
        self.chunking_size = chunking_size
        self.chunking_blur = chunking_blur

        # Hide tqdm
        self.set_progress_bar_config(disable=True)

        # Add config for xl
        self.register_to_config(
            force_full_precision_vae=force_full_precision_vae,
            requires_aesthetic_score=requires_aesthetic_score,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
        )

        # Add an image processor for later
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Register other networks
        self.register_modules(controlnet=controlnet, text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2)

    @classmethod
    def debug_tensors(cls, **kwargs: Union[Dict, List, torch.Tensor]) -> None:
        """
        Debug logs tensors.
        """
        for key in kwargs:
            value = kwargs[key]
            if isinstance(value, list):
                for i, v in enumerate(value):
                    cls.debug_tensors(**{f"{key}_{i}": v})
            elif isinstance(value, dict):
                for k in value:
                    cls.debug_tensors(**{f"{key}_{k}": value[k]})
            else:
                logger.debug(f"{key} = {value.shape} ({value.dtype})")

    @classmethod
    def from_ckpt(
        cls,
        checkpoint_path: str,
        cache_dir: str,
        prediction_type: Optional[str] = None,
        image_size: int = 512,
        scheduler_type: Literal["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm", "ddim"] = "ddim",
        vae_path: Optional[str] = None,
        load_safety_checker: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        upcast_attention: Optional[bool] = None,
        extract_ema: Optional[bool] = None,
        **kwargs: Any,
    ) -> EnfugueStableDiffusionPipeline:
        """
        Loads a checkpoint into this pipeline.
        Diffusers' `from_pretrained` lets us pass arbitrary kwargs in, but `from_ckpt` does not.
        That's why we override it for this method - most of this is copied from
        https://github.com/huggingface/diffusers/blob/49949f321d9b034440b52e54937fd2df3027bf0a/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py
        """
        logger.debug(f"Reading checkpoint file {checkpoint_path}")
        try:
            if checkpoint_path.endswith("safetensors"):
                from safetensors import safe_open

                checkpoint = {}
                with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint[key] = f.get_tensor(key)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as ex:
            # Usually a bad file
            raise IOError(f"Recevied exception reading checkpoint {checkpoint_path}, please ensure file integrity.\n{type(ex).__name__}: {ex}")

        # Sometimes models don't have the global_step item
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        else:
            global_step = None

        # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
        # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
        while "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        key_name_2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        key_name_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
        key_name_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"

        # SD v1
        config_url = (
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
        )

        if key_name_2_1 in checkpoint and checkpoint[key_name_2_1].shape[-1] == 1024:
            # SD v2.1
            config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"

            if global_step == 110000:
                # v2.1 needs to upcast attention
                upcast_attention = True
        elif key_name_xl_base in checkpoint:
            # SDXL Base
            config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
        elif key_name_xl_refiner in checkpoint:
            # SDXL Refiner
            config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

        original_config_file = check_download_to_dir(config_url, cache_dir, check_size=False)
        from omegaconf import OmegaConf

        original_config = OmegaConf.load(original_config_file)

        num_in_channels = 9 if "inpaint" in os.path.basename(checkpoint_path).lower() else 4
        if "unet_config" in original_config["model"]["params"]:  # type: ignore
            # SD 1 or 2
            original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels  # type: ignore
        elif "network_config" in original_config["model"]["params"]:  # type: ignore
            # SDXL
            original_config["model"]["params"]["network_config"]["params"]["in_channels"] = num_in_channels  # type: ignore

        if (
            "parameterization" in original_config["model"]["params"]  # type: ignore
            and original_config["model"]["params"]["parameterization"] == "v"  # type: ignore
        ):
            if prediction_type is None:
                # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
                # as it relies on a brittle global step parameter here
                prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
            if image_size is None:
                # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
                # as it relies on a brittle global step parameter here
                image_size = 512 if global_step == 875000 else 768  # type: ignore[unreachable]
        else:
            if prediction_type is None:
                prediction_type = "epsilon"
            if image_size is None:
                image_size = 512  # type: ignore[unreachable]

        model_type = None
        if (
            "cond_stage_config" in original_config.model.params
            and original_config.model.params.cond_stage_config is not None
        ):
            model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
        elif original_config.model.params.network_config is not None:
            if original_config.model.params.network_config.params.context_dim == 2048:
                model_type = "SDXL"
            else:
                model_type = "SDXL-Refiner"

        num_train_timesteps = 1000  # Default is SDXL
        if "timesteps" in original_config.model.params:
            # SD 1 or 2
            num_train_timesteps = original_config.model.params.timesteps

        if model_type in ["SDXL", "SDXL-Refiner"]:
            image_size = 1024
            scheduler_dict = {
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "interpolation_type": "linear",
                "num_train_timesteps": num_train_timesteps,
                "prediction_type": "epsilon",
                "sample_max_value": 1.0,
                "set_alpha_to_one": False,
                "skip_prk_steps": True,
                "steps_offset": 1,
                "timestep_spacing": "leading",
            }
            scheduler = EulerDiscreteScheduler.from_config(scheduler_dict)
            scheduler_type = "euler"
        else:
            beta_start = original_config.model.params.linear_start
            beta_end = original_config.model.params.linear_end
            scheduler = DDIMScheduler(
                beta_end=beta_end,
                beta_schedule="scaled_linear",
                beta_start=beta_start,
                num_train_timesteps=num_train_timesteps,
                steps_offset=1,
                clip_sample=False,
                set_alpha_to_one=False,
                prediction_type=prediction_type,
            )

        # make sure scheduler works correctly with DDIM
        scheduler.register_to_config(clip_sample=False)

        if scheduler_type == "pndm":
            config = dict(scheduler.config)
            config["skip_prk_steps"] = True
            scheduler = PNDMScheduler.from_config(config)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "dpm":
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        elif scheduler_type == "ddim":
            scheduler = scheduler
        else:
            raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

        unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
        unet_config["upcast_attention"] = upcast_attention
        unet = UNet2DConditionModel(**unet_config)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
        )

        unet.load_state_dict(converted_unet_checkpoint)

        # Convert the VAE model.
        if vae_path is None:
            try:
                vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

                if (
                    "model" in original_config
                    and "params" in original_config.model
                    and "scale_factor" in original_config.model.params
                ):
                    vae_scaling_factor = original_config.model.params.scale_factor
                else:
                    vae_scaling_factor = 0.18215  # default SD scaling factor

                vae_config["scaling_factor"] = vae_scaling_factor
                vae = AutoencoderKL(**vae_config)
                vae.load_state_dict(converted_vae_checkpoint)
            except KeyError as ex:
                default_path = "stabilityai/sdxl-vae" if model_type in ["SDXL", "SDXL-Refiner"] else "stabilityai/sd-vae-ft-ema"
                logger.error(f"Malformed VAE state dictionary detected; missing required key '{ex}'. Reverting to default model {default_path}")
                vae = AutoencoderKL.from_pretrained(default_path, cache_dir=cache_dir)
        else:
            vae = AutoencoderKL.from_pretrained(vae_path, cache_dir=cache_dir)

        if load_safety_checker:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
            feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
        else:
            safety_checker = None
            feature_extractor = None

        # Convert the text model.
        if model_type == "FrozenCLIPEmbedder":
            logger.debug("Using Stable Diffusion v1 pipeline.")
            text_model = convert_ldm_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            kwargs["text_encoder_2"] = None
            kwargs["tokenizer_2"] = None
            pipe = cls(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                **kwargs,
            )
        elif model_type == "SDXL":
            logger.debug("Using Stable Diffusion XL pipeline.")
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            text_encoder = convert_ldm_clip_checkpoint(checkpoint)
            tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", pad_token="!")

            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                prefix="conditioner.embedders.1.model.",
                has_projection=True,
                projection_dim=1280,
            )
            pipe = cls(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                force_zeros_for_empty_prompt=True,
                **kwargs,
            )
        elif model_type == "SDXL-Refiner":
            logger.debug("Using Stable Diffusion XL refiner pipeline.")
            tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", pad_token="!")
            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                prefix="conditioner.embedders.0.model.",
                has_projection=True,
                projection_dim=1280,
            )
            pipe = cls(
                vae=vae,
                text_encoder=None,
                text_encoder_2=text_encoder_2,
                tokenizer=None,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                force_zeros_for_empty_prompt=False,
                requires_aesthetic_score=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported model type {model_type}")
        if torch_dtype is not None:
            return pipe.to(torch_dtype=torch_dtype)
        return pipe

    @property
    def is_sdxl(self) -> bool:
        """
        Returns true if this is using SDXL (base or refiner)
        """
        return self.tokenizer_2 is not None and self.text_encoder_2 is not None
    
    @property
    def is_sdxl_refiner(self) -> bool:
        """
        Returns true if this is using SDXL refiner
        """
        return self.is_sdxl and self.tokenizer is None and self.text_encoder is None
    
    @property
    def is_sdxl_base(self) -> bool:
        """
        Returns true if this is using SDXL base
        """
        return self.is_sdxl and self.tokenizer is not None and self.text_encoder is not None

    @property
    def module_size(self) -> int:
        """
        Returns the size of the pipeline models in bytes.
        """
        size = 0
        for module in self.get_modules():
            module_size = 0
            for param in module.parameters():
                size += param.nelement() * param.element_size()
            for buffer in module.buffers():
                size += buffer.nelement() * buffer.element_size()
        return size
    
    def get_size_from_module(self, module: torch.nn.Module) -> int:
        """
        Gets the size of a module in bytes
        """
        size = 0
        for param in module.parameters():
            size += param.nelement() * param.element_size()
        for buffer in module.buffers():
            size += buffer.nelement() * buffer.element_size()
        return size

    def get_modules(self) -> List[torch.nn.Module]:
        """
        Gets modules in this pipeline ordered in decreasing size.
        """
        modules = []
        module_names, _ = self._get_signature_keys(self)
        for name in module_names:
            module = getattr(self, name, None)
            if isinstance(module, torch.nn.Module):
                modules.append(module)
        modules.sort(key = lambda item: self.get_size_from_module(item), reverse=True)
        return modules

    def encode_prompt(
        self,
        prompt: Optional[str],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        prompt_2: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encodes the prompt into text encoder hidden states.
        See https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if self.is_sdxl_base:
            prompts = [
                prompt if prompt else prompt_2,
                prompt_2 if prompt_2 else prompt
            ]
            negative_prompts = [
                negative_prompt if negative_prompt else negative_prompt_2,
                negative_prompt_2 if negative_prompt_2 else negative_prompt
            ]
        else:
            if prompt and prompt_2:
                logger.debug("Merging prompt and prompt_2")
                prompt = str(TokenMerger(prompt, prompt_2))
            elif not prompt and prompt_2:
                logger.debug("Using prompt_2 for empty primary prompt")
                prompt = prompt_2
            
            if negative_prompt and negative_prompt_2:
                logger.debug("Merging negative_prompt and negative_prompt_2")
                negative_prompt = str(TokenMerger(negative_prompt, negative_prompt_2))
            elif not negative_prompt and negative_prompt_2:
                logger.debug("Using negative_prompt_2 for empty primary negative_prompt")
                negative_prompt = negative_prompt_2

            prompts = [prompt, prompt]
            negative_prompts = [negative_prompt, negative_prompt]

        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        if prompt_embeds is None:
            prompt_embeds_list = []
            for tokenizer, text_encoder, prompt in zip(tokenizers, text_encoders, prompts):
                if not tokenizer or not text_encoder:
                    continue
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                    not self.is_sdxl
                    and hasattr(text_encoder.config, "use_attention_mask")
                    and text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                text_input_ids = text_input_ids.to(device=device)
                prompt_embeds = text_encoder(
                    text_input_ids, output_hidden_states=self.is_sdxl, attention_mask=attention_mask
                )

                if self.is_sdxl:
                    pooled_prompt_embeds = prompt_embeds[0]  # type: ignore
                    prompt_embeds = prompt_embeds.hidden_states[-2]  # type: ignore
                else:
                    prompt_embeds = prompt_embeds[0].to(dtype=text_encoder.dtype, device=device)  # type: ignore

                bs_embed, seq_len, _ = prompt_embeds.shape  # type: ignore
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)  # type: ignore
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                if self.is_sdxl:
                    prompt_embeds_list.append(prompt_embeds)
            if self.is_sdxl:
                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if self.is_sdxl and do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)  # type: ignore
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str] = [negative_prompt or ""]
            negative_prompt_embeds_list = []

            for tokenizer, text_encoder, negative_prompt in zip(tokenizers, text_encoders, negative_prompts):
                if tokenizer is None or text_encoder is None:
                    continue
                max_length = prompt_embeds.shape[1]  # type: ignore
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if (
                    not self.is_sdxl
                    and hasattr(text_encoder.config, "use_attention_mask")
                    and text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device), output_hidden_states=self.is_sdxl, attention_mask=attention_mask
                )

                if self.is_sdxl:
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]  # type: ignore
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]  # type: ignore
                else:
                    negative_prompt_embeds = negative_prompt_embeds[0]  # type: ignore

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]  # type: ignore

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)  # type: ignore

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(num_images_per_prompt, seq_len, -1)

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    if not self.is_sdxl:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])  # type: ignore
                if self.is_sdxl:
                    negative_prompt_embeds_list.append(negative_prompt_embeds)
            if self.is_sdxl:
                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.is_sdxl:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds  # type: ignore
        return prompt_embeds  # type: ignore

    @contextmanager
    def get_runtime_context(self, batch_size: int, device: Union[str, torch.device]) -> Iterator[None]:
        """
        Used by other implementations (tensorrt), but not base.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.unet.to(device)
        self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(device)
        if self.controlnet is not None:
            self.controlnet.to(device)
        if device.type == "cpu":
            with torch.autocast("cpu"):
                yield
        else:
            yield

    def load_lycoris_weights(
        self,
        weights_path: str,
        multiplier: float = 1.0,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """
        Loads lycoris weights using the official package
        """
        name, ext = os.path.splitext(os.path.basename(weights_path))
        if ext == ".safetensors":
            state_dict = safetensors.torch.load_file(weights_path, device="cpu")
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        while "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        from lycoris.utils import merge

        merge((self.text_encoder, self.vae, self.unet), state_dict, multiplier, device="cpu")

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        multiplier: float = 1.0,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """
        Call the appropriate adapted fix based on pipeline class
        """
        if self.is_sdxl:
            # Call SDXL fix
            return self.load_sdxl_lora_weights(
                pretrained_model_name_or_path_or_dict,
                multiplier=multiplier,
                dtype=dtype,
                **kwargs
            )
        elif (
            isinstance(pretrained_model_name_or_path_or_dict, str) and
            pretrained_model_name_or_path_or_dict.endswith(".safetensors")
        ):
            # Call safetensors fix
            return self.load_safetensors_lora_weights(
                pretrained_model_name_or_path_or_dict,
                multiplier=multiplier,
                dtype=dtype,
                **kwargs
            )
        # Return parent
        return super(EnfugueStableDiffusionPipeline, self).load_lora_weights(
            pretrained_model_name_or_path_or_dict, **kwargs
        )

    def load_sdxl_lora_weights(
        self, 
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        multiplier: float = 1.0,
        **kwargs: Any
    ) -> None:
        """
        Fix adapted from https://github.com/huggingface/diffusers/blob/4a4cdd6b07a36bbf58643e96c9a16d3851ca5bc5/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        """
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=multiplier,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=multiplier,
            )

    def load_safetensors_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        multiplier: float = 1.0,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """
        Fix adapted from here: https://github.com/huggingface/diffusers/issues/3064#issuecomment-1545013909
        """
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"

        # load LoRA weight from .safetensors
        state_dict = safetensors.torch.load_file(pretrained_model_name_or_path_or_dict, device="cpu")

        updates: Mapping[str, Any] = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split(".", 1)
            updates[layer][elem] = value

        index = 0
        # directly update weight in diffusers model
        for layer, elems in updates.items():
            index += 1

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.text_encoder
                if not curr_layer:
                    raise ValueError("No text encoder, cannot load LoRA weights.")
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.unet
            
            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems["lora_up.weight"].to(dtype)
            weight_down = elems["lora_down.weight"].to(dtype)
            alpha = elems["alpha"]
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += (
                    multiplier
                    * alpha
                    * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Denomalizes image data from [-1, 1] to [0, 1]
        """
        return (latents / 2 + 0.5).clamp(0, 1)

    def prepare_mask_and_image(
        self,
        mask: Union[np.ndarray, PIL.Image.Image, torch.Tensor],
        image: Union[np.ndarray, PIL.Image.Image, torch.Tensor],
        return_image: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Prepares a mask and image for inpainting.
        """
        if isinstance(image, torch.Tensor):
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

            # Batch single image
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            # Batch and add channel dim for single mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)

            # Batch single mask or add channel dim
            if mask.ndim == 3:
                # Single batched mask, no channel dim or single mask not batched but channel dim
                if mask.shape[0] == 1:
                    mask = mask.unsqueeze(0)

                # Batched masks no channel dim
                else:
                    mask = mask.unsqueeze(1)

            assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
            assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
            assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")

            # Check mask is in [0, 1]
            if mask.min() < 0 or mask.max() > 1:
                raise ValueError("Mask should be in [0, 1] range")

            # Binarize mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

            # Image as float32
            image = image.to(dtype=torch.float32)
        elif isinstance(mask, torch.Tensor):
            raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

            # preprocess mask
            if isinstance(mask, (PIL.Image.Image, np.ndarray)):
                mask = [mask]

            if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
                mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
                mask = mask.astype(np.float32) / 255.0
            elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
                mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

            # binarize mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)
        if return_image:
            return mask, masked_image, image
        return mask, masked_image

    def create_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Creates random latents of a particular shape and type.
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        logger.debug(f"Creating random latents of shape {shape} and type {dtype}")
        random_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return random_latents * self.scheduler.init_noise_sigma

    def encode_image_unchunked(
        self, image: torch.Tensor, dtype: torch.dtype, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Encodes an image without chunking using the VAE.
        """
        logger.debug("Encoding image (unchunked).")
        if self.config.force_full_precision_vae:
            self.vae.to(dtype=torch.float32)
            image = image.float()
        latents = self.vae.encode(image).latent_dist.sample(generator) * self.vae.config.scaling_factor
        if self.config.force_full_precision_vae:
            self.vae.to(dtype=dtype)
        return latents.to(dtype=dtype)

    def encode_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Encodes an image in chunks using the VAE.
        """
        _, _, height, width = image.shape
        chunks = self.get_chunks(height, width)
        total_steps = len(chunks)

        if total_steps == 1:
            result = self.encode_image_unchunked(image, dtype, generator)
            if progress_callback is not None:
                progress_callback()
            return result

        logger.debug(f"Encoding image in {total_steps} steps.")

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        engine_latent_size = self.engine_size // self.vae_scale_factor
        num_channels = self.vae.config.latent_channels

        count = torch.zeros((1, num_channels, latent_height, latent_width)).to(device=device)
        value = torch.zeros_like(count)

        if self.config.force_full_precision_vae:
            self.vae.to(dtype=torch.float32)
            image = image.float()
        else:
            self.vae.to(dtype=image.dtype)
        for i, (top, bottom, left, right) in enumerate(chunks):
            top_px = top * self.vae_scale_factor
            bottom_px = bottom * self.vae_scale_factor
            left_px = left * self.vae_scale_factor
            right_px = right * self.vae_scale_factor

            image_view = image[:, :, top_px:bottom_px, left_px:right_px]

            encoded_image = self.vae.encode(image_view).latent_dist.sample(generator).to(device)

            # Blur edges as needed
            multiplier = torch.ones_like(encoded_image)
            if self.chunking_blur > 0:
                chunking_blur_latent_size = self.chunking_blur // self.vae_scale_factor
                blur_left = left > 0
                blur_top = top > 0
                blur_right = right < latent_width
                blur_bottom = bottom < latent_height

                for j in range(chunking_blur_latent_size):
                    mult = (j + 1) / (chunking_blur_latent_size + 1)
                    if blur_left:
                        multiplier[:, :, :, j] *= mult
                    if blur_top:
                        multiplier[:, :, j, :] *= mult
                    if blur_right:
                        multiplier[:, :, :, engine_latent_size - j - 1] *= mult
                    if blur_bottom:
                        multiplier[:, :, engine_latent_size - j - 1, :] *= mult

            value[:, :, top:bottom, left:right] += encoded_image * multiplier
            count[:, :, top:bottom, left:right] += multiplier

            if progress_callback is not None:
                progress_callback()
        if self.config.force_full_precision_vae:
            self.vae.to(dtype=dtype)
        return (torch.where(count > 0, value / count, value) * self.vae.config.scaling_factor).to(dtype=dtype)

    def prepare_image_latents(
        self,
        image: Union[torch.Tensor, PIL.Image.Image],
        timestep: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Prepares latents from an image, adding initial noise for img2img inference
        """
        image = image.to(device=device, dtype=dtype)
        init_latents = self.encode_image(
            image, device=device, generator=generator, dtype=dtype, progress_callback=progress_callback
        )

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # duplicate images to match batch size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # add noise in accordance with timesteps
        return self.scheduler.add_noise(init_latents, noise, timestep)

    def prepare_mask_latents(
        self,
        mask: Union[PIL.Image.Image, torch.Tensor],
        image: Union[PIL.Image.Image, torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
        do_classifier_free_guidance: bool = False,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares both mask and image latents for inpainting
        """
        tensor_height = height // self.vae_scale_factor
        tensor_width = width // self.vae_scale_factor
        tensor_size = (tensor_height, tensor_width)
        mask = torch.nn.functional.interpolate(mask, size=tensor_size)
        mask = mask.to(device=device, dtype=dtype)
        image = image.to(device=device, dtype=dtype)

        latents = self.encode_image(
            image, device=device, generator=generator, dtype=dtype, progress_callback=progress_callback
        ).to(device=device)

        # duplicate mask and latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if latents.shape[0] < batch_size:
            if not batch_size % latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            latents = latents.repeat(batch_size // latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        latents = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # aligning device to prevent device errors when concating it with the latent model input
        latents = latents.to(device=device, dtype=dtype)
        return mask, latents

    def get_timesteps(
        self, num_inference_steps: int, strength: float, device: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Gets the original timesteps from the scheduler based on strength when doing img2img
        """
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def get_add_time_ids(
        self,
        original_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        target_size: Tuple[int, int],
        dtype: torch.dtype,
        aesthetic_score: Optional[float] = None,
        negative_aesthetic_score: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets added time embedding vectors for SDXL
        """
        if not self.text_encoder_2:
            raise ValueError("Missing text encoder 2, incorrect call of `get_add_time_ids` on non-SDXL pipeline.")
        if (
            aesthetic_score is not None
            and negative_aesthetic_score is not None
            and self.config.requires_aesthetic_score
        ):
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = None

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetic_score` with `pipe.register_to_config(requires_aesthetic_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetic_score` with `pipe.register_to_config(requires_aesthetic_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)  # type: ignore
        if add_neg_time_ids is not None:
            add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)  # type: ignore

        return add_time_ids, add_neg_time_ids  # type: ignore

    def predict_noise_residual(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        embeddings: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the UNet to predict noise residual.
        """
        kwargs = {}
        if added_cond_kwargs is not None:
            kwargs["added_cond_kwargs"] = added_cond_kwargs
        return self.unet(
            latents,
            timestep,
            encoder_hidden_states=embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=False,
            **kwargs,
        )[0]

    def prepare_control_image(
        self,
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        width: int,
        height: int,
        batch_size: int,
        num_images_per_prompt: int,
        device: Union[str, torch.Tensor],
        dtype: torch.dtype,
        do_classifier_free_guidance=False,
    ):
        """
        Prepares an image for controlnet conditioning.
        """
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image.to(device=device, dtype=dtype)

    def prepare_controlnet_inpaint_control_image(
        self,
        image: PIL.Image.Image,
        mask: PIL.Image.Image,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Combines the image and mask into a condition for controlnet inpainting.
        """
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == mask.shape[0:1], "image and image_mask must have the same image size"
        image[mask > 0.5] = -1.0  # set as masked pixel

        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        return image.to(device=device, dtype=dtype)

    def get_minimal_chunks(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Gets the minimum chunks that cover the shape in multiples of the engine width
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        latent_window_size = self.engine_size // self.vae_scale_factor
        horizontal_blocks = math.ceil(latent_width / latent_window_size)
        vertical_blocks = math.ceil(latent_height / latent_window_size)
        total_blocks = vertical_blocks * horizontal_blocks
        chunks = []

        for i in range(total_blocks):
            top = (i // horizontal_blocks) * latent_window_size
            bottom = top + latent_window_size

            left = (i % horizontal_blocks) * latent_window_size
            right = left + latent_window_size

            if bottom > latent_height:
                offset = bottom - latent_height
                bottom -= offset
                top -= offset
            if right > latent_width:
                offset = right - latent_width
                right -= offset
                left -= offset
            chunks.append((top, bottom, left, right))
        return chunks

    def get_chunks(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Gets the chunked latent indices for multidiffusion
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        if not self.chunking_size:
            return [(0, latent_height, 0, latent_width)]

        latent_chunking_size = self.chunking_size // self.vae_scale_factor
        latent_window_size = self.engine_size // self.vae_scale_factor

        vertical_blocks = math.ceil((latent_height - latent_window_size) / latent_chunking_size + 1)
        horizontal_blocks = math.ceil((latent_width - latent_window_size) / latent_chunking_size + 1)
        total_blocks = vertical_blocks * horizontal_blocks
        chunks = []

        for i in range(int(total_blocks)):
            top = (i // horizontal_blocks) * latent_chunking_size
            bottom = top + latent_window_size
            left = (i % horizontal_blocks) * latent_chunking_size
            right = left + latent_window_size

            if bottom > latent_height:
                offset = bottom - latent_height
                bottom -= offset
                top -= offset
            if right > latent_width:
                offset = right - latent_width
                right -= offset
                left -= offset

            chunks.append((top, bottom, left, right))

        return chunks

    def get_controlnet_conditioning_blocks(
        self,
        device: Union[str, torch.device],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor],
        conditioning_scale: float,
        added_cond_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Executes the controlnet
        """
        if controlnet_cond is None or self.controlnet is None:
            return None, None

        return self.controlnet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

    def denoise_unchunked(
        self,
        height: int,
        width: int,
        device: Union[str, torch.device],
        num_inference_steps: int,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        do_classifier_free_guidance: bool = False,
        is_inpainting_unet: bool = False,
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        latent_callback: Optional[Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
        conditioning_scale: float = 1.0,
        extra_step_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Executes the denoising loop without chunking.
        """
        if extra_step_kwargs is None:
            extra_step_kwargs = {}

        num_steps = len(timesteps)
        num_warmup_steps = num_steps - num_inference_steps * self.scheduler.order
        
        noise = None
        if mask is not None and mask_image is not None and not is_inpainting_unet:
            noise = latents.detach().clone() / self.scheduler.init_noise_sigma
            noise = noise.to(device=device)

        logger.debug(f"Denoising image in {num_steps} steps on {device} (unchunked)")

        steps_since_last_callback = 0
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Get controlnet input if configured
            if control_image is not None:
                down_block, mid_block = self.get_controlnet_conditioning_blocks(
                    device=device,
                    latents=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=conditioning_scale,
                    added_cond_kwargs=added_cond_kwargs,
                )
            else:
                down_block, mid_block = None, None

            # add other dimensions to unet input if set
            if mask is not None and mask_image is not None and is_inpainting_unet:
                latent_model_input = torch.cat(
                    [latent_model_input, mask, mask_image],
                    dim=1,
                )

            # predict the noise residual
            noise_pred = self.predict_noise_residual(
                latent_model_input,
                t,
                prompt_embeds,
                cross_attention_kwargs,
                added_cond_kwargs,
                down_block,
                mid_block,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
            ).prev_sample

            # If using mask and not using fine-tuned inpainting, then we calculate
            # the same denoising on the image without unet and cross with the
            # calculated unet input * mask
            if mask is not None and image is not None and not is_inpainting_unet:
                init_latents = image[:1]
                init_mask = mask[:1]

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents = self.scheduler.add_noise(
                        init_latents,
                        noise,
                        torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents + init_mask * latents

            if progress_callback is not None:
                progress_callback()

            # call the callback, if provided
            steps_since_last_callback += 1
            if (
                latent_callback is not None
                and latent_callback_steps is not None
                and steps_since_last_callback >= latent_callback_steps
                and (i == num_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0))
            ):
                steps_since_last_callback = 0
                latent_callback_value = latents

                if latent_callback_type != "latent":
                    latent_callback_value = self.decode_latents(
                        latent_callback_value, device=device, progress_callback=progress_callback
                    )
                    latent_callback_value = self.denormalize_latents(latent_callback_value)
                    if latent_callback_type != "pt":
                        latent_callback_value = self.image_processor.pt_to_numpy(latent_callback_value)
                        if latent_callback_type == "pil":
                            latent_callback_value = self.image_processor.numpy_to_pil(latent_callback_value)
                latent_callback(latent_callback_value)

        return latents

    def get_scheduler_state(self) -> Dict[str, Any]:
        """
        Gets the state dictionary from the current scheduler.
        Copies it in a safe way.
        """
        data: Dict[str, Any] = {}
        scheduler_data = self.scheduler.__dict__
        for key in scheduler_data:
            try:
                data[key] = copy.deepcopy(scheduler_data[key])
            except:
                pass
        return data


    def denoise(
        self,
        height: int,
        width: int,
        device: Union[str, torch.device],
        num_inference_steps: int,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        do_classifier_free_guidance: bool = False,
        is_inpainting_unet: bool = False,
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        latent_callback: Optional[Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
        conditioning_scale: float = 1.0,
        extra_step_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Executes the denoising loop.
        """
        if extra_step_kwargs is None:
            extra_step_kwargs = {}

        chunks = self.get_chunks(height, width)
        num_chunks = len(chunks)

        if num_chunks <= 1:
            return self.denoise_unchunked(
                height=height,
                width=width,
                device=device,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                latents=latents,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
                is_inpainting_unet=is_inpainting_unet,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mask=mask,
                mask_image=mask_image,
                image=image,
                control_image=control_image,
                conditioning_scale=conditioning_scale,
                progress_callback=progress_callback,
                latent_callback=latent_callback,
                latent_callback_steps=latent_callback_steps,
                latent_callback_type=latent_callback_type,
                extra_step_kwargs=extra_step_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
            )

        chunk_scheduler_status = [self.get_scheduler_state()] * num_chunks
        num_steps = len(timesteps)
        num_warmup_steps = num_steps - num_inference_steps * self.scheduler.order

        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor
        engine_latent_size = self.engine_size // self.vae_scale_factor

        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        total_num_steps = num_steps * num_chunks
        logger.debug(
            f"Denoising image in {total_num_steps} steps on {device} ({num_inference_steps} inference steps * {num_chunks} chunks)"
        )

        noise = None
        if mask is not None and mask_image is not None and not is_inpainting_unet:
            noise = latents.detach().clone() / self.scheduler.init_noise_sigma
            noise = noise.to(device=device)

        steps_since_last_callback = 0
        for i, t in enumerate(timesteps):
            # zero view latents
            count.zero_()
            value.zero_()

            # iterate over chunks
            for j, (top, bottom, left, right) in enumerate(chunks):
                # Wrap IndexError to give a nice error about MultiDiff w/ some schedulers
                try:
                    # Get pixel indices
                    top_px = top * self.vae_scale_factor
                    bottom_px = bottom * self.vae_scale_factor
                    left_px = left * self.vae_scale_factor
                    right_px = right * self.vae_scale_factor

                    # Slice latents
                    latents_for_view = latents[:, :, top:bottom, left:right]

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_for_view] * 2) if do_classifier_free_guidance else latents_for_view
                    )

                    # Re-match chunk scheduler status
                    self.scheduler.__dict__.update(chunk_scheduler_status[j])

                    # Scale model input
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # Get controlnet input if configured
                    if control_image is not None:
                        down_block, mid_block = self.get_controlnet_conditioning_blocks(
                            device=device,
                            latents=latent_model_input,
                            timestep=t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_cond=control_image[:, :, top_px:bottom_px, left_px:right_px],
                            conditioning_scale=conditioning_scale,
                            added_cond_kwargs=added_cond_kwargs
                        )
                    else:
                        down_block, mid_block = None, None

                    # add other dimensions to unet input if set
                    if mask is not None and mask_image is not None and is_inpainting_unet:
                        latent_model_input = torch.cat(
                            [
                                latent_model_input,
                                mask[:, :, top:bottom, left:right],
                                mask_image[:, :, top:bottom, left:right],
                            ],
                            dim=1,
                        )

                    # predict the noise residual
                    noise_pred = self.predict_noise_residual(
                        latents=latent_model_input,
                        timestep=t,
                        embeddings=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down_block,
                        mid_block_additional_residual=mid_block,
                    )

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    denoised_latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents_for_view,
                        **extra_step_kwargs,
                    ).prev_sample
                except IndexError:
                    raise RuntimeError(f"Received IndexError during denoising. It is likely that the scheduler you are using ({type(self.scheduler).__name__}) does not work with Multi-Diffusion, and you should avoid using this when chunking is enabled.")

                # Save chunk scheduler status after sample
                chunk_scheduler_status[j] = self.get_scheduler_state()

                # If using mask and not using fine-tuned inpainting, then we calculate
                # the same denoising on the image without unet and cross with the
                # calculated unet input * mask
                if mask is not None and image is not None and noise is not None and not is_inpainting_unet:
                    init_latents = (image[:, :, top:bottom, left:right])[:1]
                    init_mask = (mask[:, :, top:bottom, left:right])[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents = self.scheduler.add_noise(
                            init_latents,
                            noise[:, :, top:bottom, left:right],
                            torch.tensor([noise_timestep])
                        )

                    denoised_latents = (1 - init_mask) * init_latents + init_mask * denoised_latents

                # Blur edges as needed
                multiplier = torch.ones_like(denoised_latents)
                if self.chunking_blur > 0:
                    chunking_blur_latent_size = self.chunking_blur // self.vae_scale_factor
                    blur_left = left > 0
                    blur_top = top > 0
                    blur_right = right < latent_width
                    blur_bottom = bottom < latent_height

                    for j in range(chunking_blur_latent_size):
                        mult = (j + 1) / (chunking_blur_latent_size + 1)
                        if blur_left:
                            multiplier[:, :, :, j] *= mult
                        if blur_top:
                            multiplier[:, :, j, :] *= mult
                        if blur_right:
                            multiplier[:, :, :, engine_latent_size - j - 1] *= mult
                        if blur_bottom:
                            multiplier[:, :, engine_latent_size - j - 1, :] *= mult

                value[:, :, top:bottom, left:right] += denoised_latents * multiplier
                count[:, :, top:bottom, left:right] += multiplier

                # Call the progress callback
                if progress_callback is not None:
                    progress_callback()

            # multidiffusion
            latents = torch.where(count > 0, value / count, value)

            # call the latent_callback, if provided
            steps_since_last_callback += 1
            if (
                latent_callback is not None
                and latent_callback_steps is not None
                and steps_since_last_callback >= latent_callback_steps
                and (i == num_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0))
            ):
                steps_since_last_callback = 0
                latent_callback_value = latents

                if latent_callback_type != "latent":
                    latent_callback_value = self.decode_latents(
                        latent_callback_value, device=device, progress_callback=progress_callback
                    )
                    latent_callback_value = self.denormalize_latents(latent_callback_value)
                    if latent_callback_type != "pt":
                        latent_callback_value = self.image_processor.pt_to_numpy(latent_callback_value)
                        if latent_callback_type == "pil":
                            latent_callback_value = self.image_processor.numpy_to_pil(latent_callback_value)
                latent_callback(latent_callback_value)

        return latents

    def decode_latent_view(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Issues the command to decode a chunk of latents with the VAE.
        """
        return self.vae.decode(latents, return_dict=False)[0]

    def decode_latents_unchunked(self, latents: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        """
        Decodes the latents using the VAE without chunking.
        """
        return self.decode_latent_view(latents).to(device=device)

    def decode_latents(
        self,
        latents: torch.Tensor,
        device: Union[str, torch.device],
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Decodes the latents in chunks as necessary.
        """
        samples, _, height, width = latents.shape
        height *= self.vae_scale_factor
        width *= self.vae_scale_factor

        latents = 1 / self.vae.config.scaling_factor * latents

        chunks = self.get_chunks(height, width)
        total_steps = len(chunks)

        revert_dtype = None

        if self.config.force_full_precision_vae:
            # Resist overflow
            revert_dtype = latents.dtype
            self.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)

        if total_steps <= 1:
            result = self.decode_latents_unchunked(latents, device)
            if progress_callback is not None:
                progress_callback()
            if self.config.force_full_precision_vae:
                self.vae.to(dtype=latents.dtype)
            return result

        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor

        count = torch.zeros((samples, 3, height, width)).to(device=device, dtype=latents.dtype)
        value = torch.zeros_like(count)

        logger.debug(f"Decoding latents in {total_steps} steps")

        # iterate over chunks
        for i, (top, bottom, left, right) in enumerate(chunks):
            # Slice latents
            latents_for_view = latents[:, :, top:bottom, left:right]

            # Get pixel indices
            top_px = top * self.vae_scale_factor
            bottom_px = bottom * self.vae_scale_factor
            left_px = left * self.vae_scale_factor
            right_px = right * self.vae_scale_factor

            # Decode latents
            decoded_latents = self.decode_latent_view(latents_for_view).to(device=device)

            # Blur edges as needed
            multiplier = torch.ones_like(decoded_latents)
            if self.chunking_blur > 0:
                chunking_blur_size = self.chunking_blur
                blur_left = left > 0
                blur_top = top > 0
                blur_right = right < latent_width
                blur_bottom = bottom < latent_height

                for j in range(chunking_blur_size):
                    mult = (j + 1) / (chunking_blur_size + 1)
                    if blur_left:
                        multiplier[:, :, :, j] *= mult
                    if blur_top:
                        multiplier[:, :, j, :] *= mult
                    if blur_right:
                        multiplier[:, :, :, self.engine_size - j - 1] *= mult
                    if blur_bottom:
                        multiplier[:, :, self.engine_size - j - 1, :] *= mult

            value[:, :, top_px:bottom_px, left_px:right_px] += decoded_latents * multiplier
            count[:, :, top_px:bottom_px, left_px:right_px] += multiplier

            if progress_callback is not None:
                progress_callback()

        # re-average pixels
        latents = torch.where(count > 0, value / count, value)
        if revert_dtype is not None:
            latents = latents.to(dtype=revert_dtype)
            self.vae.to(dtype=revert_dtype)
        return latents

    def prepare_extra_step_kwargs(
        self, generator: Optional[torch.Generator], eta: float
    ) -> Dict[str, Any]:
        """
        Prepares arguments to add during denoising
        """
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs: Dict[str, Any] = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_step_complete_callback(
        self,
        overall_steps: int,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        log_interval: int = 10,
        log_sampling_duration: Union[int, float] = 5,
    ) -> Callable[[], None]:
        """
        Creates a scoped callback to trigger during iterations
        """
        overall_step, window_start_step, its = 0, 0, 0.0
        window_start = datetime.datetime.now()
        digits = math.ceil(math.log10(overall_steps))

        def step_complete() -> None:
            nonlocal overall_step, window_start, window_start_step, its
            overall_step += 1
            if overall_step % log_interval == 0 or overall_step == overall_steps:
                seconds_in_window = (datetime.datetime.now() - window_start).total_seconds()
                its = (overall_step - window_start_step) / seconds_in_window
                unit = "s/it" if its < 1 else "it/s"
                its_display = 1 / its if its < 1 else its
                logger.debug(
                    f"{{0:0{digits}d}}/{{1:0{digits}d}}: {{2:0.2f}} {{3:s}}".format(
                        overall_step, overall_steps, its_display, unit
                    )
                )

                if seconds_in_window > log_sampling_duration:
                    window_start_step = overall_step
                    window_start = datetime.datetime.now()

            if progress_callback is not None:
                progress_callback(overall_step, overall_steps, its)

        return step_complete

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, str]] = None,
        mask: Optional[Union[PIL.Image.Image, str]] = None,
        control_image: Optional[Union[PIL.Image.Image, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        chunking_size: Optional[int] = None,
        chunking_blur: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Literal["latent", "pt", "np", "pil"] = "pil",
        return_dict: bool = True,
        scale_image: bool = True,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        latent_callback: Optional[Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
    ) -> Union[
        StableDiffusionPipelineOutput,
        Tuple[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]], Optional[List[bool]]],
    ]:
        """
        Invokes the pipeline.
        """
        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Training details
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # Allow overridding 'chunk'
        if chunking_size is not None:
            self.chunking_size = chunking_size
        if chunking_blur is not None:
            self.chunking_blur = chunking_blur
        
        # Check latent callback steps, disable if 0
        if latent_callback_steps == 0:
            latent_callback_steps = None

        # Convenient bool for later
        decode_intermediates = latent_callback_steps is not None and latent_callback is not None

        # Define outputs here to process later
        prepared_latents: Optional[torch.Tensor] = None
        prepared_control_image: Optional[torch.Tensor] = None
        output_nsfw: Optional[List[bool]] = None

        # Determine dimensionality
        is_inpainting_unet = self.unet.config.in_channels == 9

        # 2. Define call parameters
        if prompt is not None:
            batch_size = 1
        elif prompt_embeds:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Prompt or prompt embeds are required.")

        if is_inpainting_unet:
            if not image:
                logger.warning("No image present, but using inpainting model. Adding blank image.")
                image = PIL.Image.new("RGB", (width, height))
            if not mask:
                logger.warning("No mask present, but using inpainting model. Adding blank mask.")
                mask = PIL.Image.new("RGB", (width, height), (255, 255, 255))

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Calculate chunks
        num_chunks = max(1, len(self.get_chunks(height, width)))
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        if image and not mask:
            # Scale timesteps by strength
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        else:
            timesteps = self.scheduler.timesteps

        batch_size *= num_images_per_prompt
        num_scheduled_inference_steps = len(timesteps)

        # Calculate total steps including all unet and vae calls
        encoding_steps = 0
        decoding_steps = 1

        if image is not None:
            encoding_steps += 1
        if mask is not None:
            encoding_steps += 1
            if not is_inpainting_unet:
                encoding_steps += 1
        if decode_intermediates:
            decoding_steps += num_scheduled_inference_steps // latent_callback_steps # type: ignore

        chunk_plural = "s" if num_chunks != 1 else ""
        step_plural = "s" if num_scheduled_inference_steps != 1 else ""
        encoding_plural = "s" if encoding_steps != 1 else ""
        decoding_plural = "s" if decoding_steps != 1 else ""
        overall_num_steps = num_chunks * (num_scheduled_inference_steps + encoding_steps + decoding_steps)
        logger.debug(
            " ".join([
                f"Calculated overall steps to be {overall_num_steps}",
                f"[{num_chunks} chunk{chunk_plural} * ({num_scheduled_inference_steps} inference step{step_plural}",
                f"+ {encoding_steps} encoding step{encoding_plural} + {decoding_steps} decoding step{decoding_plural})]"
            ])
        )
        step_complete = self.get_step_complete_callback(overall_num_steps, progress_callback)
                
        if self.config.force_full_precision_vae:
            logger.debug(f"Configuration indicates VAE must be used in full precision")
            # make sure the VAE is in float32 mode, as it overflows in float16
            self.vae.to(dtype=torch.float32)
        elif self.is_sdxl:
            logger.debug(f"Configuration indicates VAE may operate in half precision")
            self.vae.to(dtype=torch.float16)

        with self.get_runtime_context(batch_size, device):
            # Base runtime has no context, but extensions do
            if self.is_sdxl:
                # XL uses more inputs for prompts than 1.5
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    prompt_2=prompt_2,
                    negative_prompt_2=negative_prompt_2
                )
            else:
                prompt_embeds = self.encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    prompt_2=prompt_2,
                    negative_prompt_2=negative_prompt_2
                )  # type: ignore
                pooled_prompt_embeds = None
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Open images if they're files
            if image and isinstance(image, str):
                image = PIL.Image.open(image)

            if mask and isinstance(mask, str):
                mask = PIL.Image.open(mask)

            # Scale images if requested
            if scale_image and image:
                image_width, image_height = image.size
                if image_width != width or image_height != height:
                    logger.debug(f"Resizing input image from {image_width}x{image_height} to {width}x{height}")
                    image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])

            if scale_image and mask:
                mask_width, mask_height = mask.size
                if mask_width != width or mask_height != height:
                    logger.debug(f"Resizing input mask from {mask_width}x{mask_height} to {width}x{height}")
                    mask = mask.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])

            # Remove any alpha mask on image, convert mask to grayscale
            if image:
                image = image.convert("RGB")
            if mask:
                mask = mask.convert("L")

            if image and mask:
                if is_inpainting_unet:
                    prepared_mask, prepared_image = self.prepare_mask_and_image(mask, image, False) # type: ignore
                    init_image = None
                else:
                    prepared_mask, prepared_image, init_image = self.prepare_mask_and_image(mask, image, True) # type: ignore
            elif image:
                prepared_image = self.image_processor.preprocess(image)
                prepared_mask, prepared_image_latents, init_image = None, None, None
            else:
                prepared_image, prepared_mask, prepared_image_latents, init_image = None, None, None, None

            if width < self.engine_size or height < self.engine_size:
                # Disable chunking
                logger.debug(f"{width}x{height} is smaller than is chunkable, disabling.")
                self.chunking_size = 0

            prompt_embeds = cast(torch.Tensor, prompt_embeds)
            if prepared_image is not None and prepared_mask is not None:
                # Inpainting
                num_channels_latents = self.vae.config.latent_channels

                if latents:
                    prepared_latents = latents.to(device) * self.schedule.init_noise_sigma
                else:
                    prepared_latents = self.create_latents(
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                    )

                prepared_mask, prepared_image_latents = self.prepare_mask_latents(
                    prepared_mask,
                    prepared_image,
                    batch_size,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    progress_callback=step_complete,
                )

                if init_image is not None:
                    init_image = init_image.to(device=device, dtype=prompt_embeds.dtype)
                    init_image = self.encode_image(
                        init_image,
                        device=device,
                        dtype=prompt_embeds.dtype,
                        generator=generator
                    )

                # prepared_latents = noise or init latents + noise
                # prepared_mask = only mask
                # prepared_image_latents = masked image
                # init_image = only image when not using inpainting unet
            elif prepared_image is not None:
                # img2img
                prepared_latents = self.prepare_image_latents(
                    prepared_image,
                    timesteps[:1].repeat(batch_size),
                    batch_size,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    progress_callback=step_complete,
                )
                # prepared_latents = img + noise
            elif latents:
                prepared_latents = latents.to(device) * self.scheduler.init_noise_sigma
                # prepared_latents = passed latents + noise
            else:
                # txt2img
                prepared_latents = self.create_latents(
                    batch_size,
                    self.unet.config.in_channels,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )
                # prepared_latents = noise

            # Look for controlnet and conditioning image
            if control_image is not None:
                if self.controlnet is None:
                    logger.warning("Control image passed, but no controlnet present. Ignoring.")
                    prepared_control_image = None
                else:
                    if type(control_image) is str:
                        control_image = PIL.Image.open(control_image)
                    self.controlnet = self.controlnet.to(device=device)  # Cast to device
                    prepared_control_image = self.prepare_control_image(
                        image=control_image,
                        height=height,
                        width=width,
                        batch_size=batch_size,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )
            elif self.controlnet is not None:
                if image and mask:
                    logger.info("Assuming controlnet inpaint, creating conditioning image from image and mask")
                    prepared_control_image = self.prepare_controlnet_inpaint_control_image(
                        image=image, mask=image, device=device, dtype=self.controlnet.dtype
                    )
                else:
                    self.controlnet = self.controlnet.to("cpu")
                    logger.warning("Controlnet present, but no conditioning image. Disabling controlnet.")

            # Should no longer be None
            prepared_latents = cast(torch.Tensor, prepared_latents)

            # 6. Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # Swap out VAE here if not needed for mostly free VRAM
            if not decode_intermediates:
                logger.debug("Intermediates are disabled, sending VAE to CPU during denoising")
                self.vae.to("cpu")

            # Make sure controlnet on device
            if self.controlnet is not None:
                self.controlnet = self.controlnet.to(device=device)

            # 7. Prepared added time IDs and embeddings (SDXL)
            if self.is_sdxl:
                negative_prompt_embeds = cast(torch.Tensor, negative_prompt_embeds)
                pooled_prompt_embeds = cast(torch.Tensor, pooled_prompt_embeds)
                negative_pooled_prompt_embeds = cast(torch.Tensor, negative_pooled_prompt_embeds)
                add_text_embeds = pooled_prompt_embeds
                if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                if self.config.requires_aesthetic_score:
                    add_time_ids, add_neg_time_ids = self.get_add_time_ids(
                        original_size=original_size,
                        crops_coords_top_left=crops_coords_top_left,
                        target_size=target_size,
                        dtype=prompt_embeds.dtype,
                        aesthetic_score=aesthetic_score,
                        negative_aesthetic_score=negative_aesthetic_score,
                    )
                    if do_classifier_free_guidance:
                        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
                else:
                    add_time_ids, _ = self.get_add_time_ids(
                        original_size=original_size,
                        crops_coords_top_left=crops_coords_top_left,
                        target_size=target_size,
                        dtype=prompt_embeds.dtype,
                    )
                    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                add_text_embeds = add_text_embeds.to(device)
                add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            else:
                added_cond_kwargs = None

            # 8. Denoising loop
            prepared_latents = self.denoise(
                height=height,
                width=width,
                device=device,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                latents=prepared_latents,
                prompt_embeds=prompt_embeds,
                conditioning_scale=conditioning_scale,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                is_inpainting_unet=is_inpainting_unet,
                mask=prepared_mask,
                mask_image=prepared_image_latents,
                image=init_image,
                control_image=prepared_control_image,
                progress_callback=step_complete,
                latent_callback=latent_callback,
                latent_callback_steps=latent_callback_steps,
                latent_callback_type=latent_callback_type,
                extra_step_kwargs=extra_step_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
            )

            # Clear no longer needed tensors
            del prepared_mask
            del prepared_image_latents

            if self.controlnet is not None:
                # Unload controlnet from GPU to save memory for decoding
                self.controlnet = self.controlnet.to("cpu")
                del prepared_control_image
            
            # Swap VAE back in here if it was not needed during denoising
            if not decode_intermediates:
                logger.debug("Reloading VAE from CPU")
                self.vae.to(
                    dtype=torch.float32 if self.config.force_full_precision_vae else prepared_latents.dtype,
                    device=device
                )
            
            if output_type != "latent":
                if self.is_sdxl:
                    use_torch_2_0_or_xformers = self.vae.decoder.mid_block.attentions[0].processor in [
                        AttnProcessor2_0,
                        XFormersAttnProcessor,
                        LoRAXFormersAttnProcessor,
                        LoRAAttnProcessor2_0,
                    ]
                    # if xformers or torch_2_0 is used attention block does not need
                    # to be in float32 which can save lots of memory
                    if not use_torch_2_0_or_xformers:
                        self.vae.post_quant_conv.to(prepared_latents.dtype)
                        self.vae.decoder.conv_in.to(prepared_latents.dtype)
                        self.vae.decoder.mid_block.to(prepared_latents.dtype)
                    else:
                        prepared_latents = prepared_latents.float()

                prepared_latents = self.decode_latents(prepared_latents, device=device, progress_callback=step_complete)

                if self.config.force_full_precision_vae:
                    self.vae.to(dtype=prepared_latents.dtype)
        if output_type == "latent":
            output = prepared_latents
        else:
            output = self.denormalize_latents(prepared_latents)
            if output_type != "pt":
                output = self.image_processor.pt_to_numpy(output)
                output_nsfw = self.run_safety_checker(output, device, prompt_embeds.dtype)[1]
                if output_type == "pil":
                    output = self.image_processor.numpy_to_pil(output)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (output, output_nsfw)

        return StableDiffusionPipelineOutput(images=output, nsfw_content_detected=output_nsfw)
