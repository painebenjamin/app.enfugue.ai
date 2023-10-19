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
    TypedDict,
    cast,
    TYPE_CHECKING
)
from typing_extensions import NotRequired

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
from omegaconf import OmegaConf

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
from diffusers.models import (
    AutoencoderKL,
    AutoencoderTiny,
    UNet2DConditionModel,
    ControlNetModel,
)
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

from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor

from pibble.util.files import load_json

from enfugue.util import logger, check_download, check_download_to_dir, TokenMerger
from enfugue.diffusion.util import (
    load_state_dict,
    empty_cache,
    make_noise,
    blend_latents,
    LatentScaler,
    MaskWeightBuilder
)
if TYPE_CHECKING:
    from enfugue.diffusers.support.ip import IPAdapter
    from enfugue.diffusion.constants import (
        MASK_TYPE_LITERAL,
        NOISE_METHOD_LITERAL,
        LATENT_BLEND_METHOD_LITERAL
    )

# This is ~64k×64k. Absurd, but I don't judge
PIL.Image.MAX_IMAGE_PIXELS = 2**32

# IP image accepted arguments
class ImagePromptArgDict(TypedDict):
    image: Union[str, PIL.Image.Image]
    scale: NotRequired[float]

ImagePromptType = Union[
    Union[str, PIL.Image.Image], # Image
    Tuple[Union[str, PIL.Image.Image], float], # Image, Scale
    ImagePromptArgDict
]
ImagePromptArgType = Optional[Union[ImagePromptType, List[ImagePromptType]]]

# Control image accepted arguments
class ControlImageArgDict(TypedDict):
    image: Union[str, PIL.Image.Image]
    scale: NotRequired[float]
    start: NotRequired[float]
    end: NotRequired[float]

ControlImageType = Union[
    Union[str, PIL.Image.Image], # Image
    Tuple[Union[str, PIL.Image.Image], float], # Image, Scale
    Tuple[Union[str, PIL.Image.Image], float, float], # Image, Scale, End
    Tuple[Union[str, PIL.Image.Image], float, float, float], # Image, Scale, Start, End
    ControlImageArgDict
]

ControlImageArgType = Optional[Dict[str, Union[ControlImageType, List[ControlImageType]]]]
PreparedControlImageArgType = Optional[Dict[str, List[Tuple[torch.Tensor, float, Optional[float], Optional[float]]]]]

class EnfugueStableDiffusionPipeline(StableDiffusionPipeline):
    """
    This pipeline merges all of the following:
    1. txt2img
    2. img2img
    3. inpainting/outpainting
    4. controlnet
    5. tensorrt
    """

    controlnets: Optional[Dict[str, ControlNetModel]]
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers
    vae: AutoencoderKL
    vae_preview: AutoencoderTiny
    tokenizer: Optional[CLIPTokenizer]
    tokenizer_2: Optional[CLIPTokenizer]
    text_encoder: Optional[CLIPTextModel]
    text_encoder_2: Optional[CLIPTextModelWithProjection]
    vae_scale_factor: int
    safety_checker: StableDiffusionSafetyChecker
    config: OmegaConf

    def __init__(
        self,
        vae: AutoencoderKL,
        vae_preview: AutoencoderTiny,
        text_encoder: Optional[CLIPTextModel],
        text_encoder_2: Optional[CLIPTextModelWithProjection],
        tokenizer: Optional[CLIPTokenizer],
        tokenizer_2: Optional[CLIPTokenizer],
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: CLIPImageProcessor,
        controlnets: Optional[Dict[str, ControlNetModel]] = None,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetic_score: bool = False,
        force_full_precision_vae: bool = False,
        ip_adapter: Optional[IPAdapter] = None,
        engine_size: int = 512,
        chunking_size: int = 64,
        chunking_mask_type: MASK_TYPE_LITERAL = "bilinear",
        chunking_mask_kwargs: Dict[str, Any] = {}
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
        self.scheduler_config = {**dict(scheduler.config)} # type: ignore[attr-defined]

        # Enfugue engine settings
        self.engine_size = engine_size
        self.chunking_size = chunking_size
        self.chunking_mask_type = chunking_mask_type
        self.chunking_mask_kwargs = chunking_mask_kwargs

        # Hide tqdm
        self.set_progress_bar_config(disable=True) # type: ignore[attr-defined]

        # Add config for xl
        self.register_to_config( # type: ignore[attr-defined]
            force_full_precision_vae=force_full_precision_vae,
            requires_aesthetic_score=requires_aesthetic_score,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
        )

        # Add an image processor for later
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Register other networks
        self.register_modules( # type: ignore[attr-defined]
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae_preview=vae_preview
        )

        # Add ControlNet map
        self.controlnets = controlnets

        # Add IP Adapter
        self.ip_adapter = ip_adapter
        self.ip_adapter_loaded = False

        # Create helpers
        self.latent_scaler = LatentScaler()

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
    def create_unet(
        cls,
        config: Dict[str, Any],
        cache_dir: str,
        is_sdxl: bool,
        is_inpainter: bool,
        **kwargs: Any
    ) -> UNet2DConditionModel:
        """
        Instantiates the UNet from config
        """
        if is_sdxl and is_inpainter:
            config_url = "https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/raw/main/unet/config.json"
            config_path = check_download_to_dir(config_url, cache_dir, check_size=False)
            config = load_json(config_path)
        return UNet2DConditionModel.from_config(config)

    @classmethod
    def from_ckpt(
        cls,
        checkpoint_path: str,
        cache_dir: str,
        prediction_type: Optional[str] = None,
        image_size: int = 512,
        scheduler_type: Literal["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm", "ddim"] = "ddim",
        vae_path: Optional[str] = None,
        vae_preview_path: Optional[str] = None,
        load_safety_checker: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        upcast_attention: Optional[bool] = None,
        extract_ema: Optional[bool] = None,
        offload_models: bool = False,
        is_inpainter=False,
        task_callback: Optional[Callable[[str], None]]=None,
        **kwargs: Any,
    ) -> EnfugueStableDiffusionPipeline:
        """
        Loads a checkpoint into this pipeline.
        Diffusers' `from_pretrained` lets us pass arbitrary kwargs in, but `from_ckpt` does not.
        That's why we override it for this method - most of this is copied from
        https://github.com/huggingface/diffusers/blob/49949f321d9b034440b52e54937fd2df3027bf0a/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py
        """
        logger.debug(f"Reading checkpoint file {checkpoint_path}")
        checkpoint = load_state_dict(checkpoint_path)

        # Sometimes models don't have the global_step item
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        else:
            global_step = None

        # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
        # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
        while "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"] # type: ignore[assignment]

        checkpoint = cast(Mapping[str, torch.Tensor], checkpoint) # type: ignore[assignment]

        # Check for config in same directory as checkpoint
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        ckpt_name, _ = os.path.splitext(os.path.basename(checkpoint_path))

        original_config_file: Optional[str] = os.path.join(ckpt_dir, f"{ckpt_name}.yaml")
        if os.path.exists(original_config_file): # type: ignore[arg-type]
            logger.info(f"Found configuration file alongside checkpoint, using {ckpt_name}.yaml")
        else:
            original_config_file = os.path.join(ckpt_dir, f"{ckpt_name}.json")
            if os.path.exists(original_config_file):
                logger.info(f"Found configuration file alongside checkpoint, using {ckpt_name}.json")
            else:
                original_config_file = None

        if original_config_file is None:
            key_name_2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            key_name_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
            key_name_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"

            # SD v1 default
            config_url = (
                "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
            )

            if key_name_2_1 in checkpoint and checkpoint[key_name_2_1].shape[-1] == 1024: # type: ignore[union-attr]
                # SD v2.1
                config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
                logger.info(f"No configuration file found for checkpoint {ckpt_name}, using Stable Diffusion V2.1")

                if global_step == 110000:
                    # v2.1 needs to upcast attention
                    upcast_attention = True
            elif key_name_xl_base in checkpoint:
                # SDXL Base
                logger.info(f"No configuration file found for checkpoint {ckpt_name}, using Stable Diffusion XL Base")
                config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
            elif key_name_xl_refiner in checkpoint:
                # SDXL Refiner
                logger.info(f"No configuration file found for checkpoint {ckpt_name}, using Stable Diffusion XL Refiner")
                config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"
            else:
                # SD v1
                logger.info(f"No configuration file found for checkpoint {ckpt_name}, using Stable Diffusion 1.5")
            original_config_file = check_download_to_dir(config_url, cache_dir, check_size=False)

        original_config = OmegaConf.load(original_config_file) # type: ignore

        if "model.diffusion_model.input_blocks.0.0.weight" in checkpoint:
            num_in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1] # type: ignore[union-attr]
            logger.info(f"Checkpoint has {num_in_channels} input channels")
        else:
            num_in_channels = 9 if is_inpainter else 4
            logger.info(f"Could not automatically determine input channels, forcing {num_in_channels} input channels")

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

        is_sdxl = isinstance(model_type, str) and model_type.startswith("SDXL")

        num_train_timesteps = 1000  # Default is SDXL
        if "timesteps" in original_config.model.params:
            # SD 1 or 2
            num_train_timesteps = original_config.model.params.timesteps

        if is_sdxl:
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

        unet = cls.create_unet(
            unet_config,
            cache_dir=cache_dir,
            is_sdxl=isinstance(model_type, str) and model_type.startswith("SDXL"),
            is_inpainter=is_inpainter,
        )

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint,
            unet_config,
            path=checkpoint_path,
            extract_ema=extract_ema
        )

        unet.load_state_dict(converted_unet_checkpoint)

        if offload_models:
            unet.to("cpu")
            empty_cache()

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
                    vae_scale_factor = original_config.model.params.scale_factor
                else:
                    vae_scale_factor = 0.18215  # default SD scaling factor

                vae_config["scaling_factor"] = vae_scale_factor
                vae = AutoencoderKL(**vae_config)
                vae.load_state_dict(converted_vae_checkpoint)
            except KeyError as ex:
                default_path = "stabilityai/sdxl-vae" if model_type in ["SDXL", "SDXL-Refiner"] else "stabilityai/sd-vae-ft-ema"
                logger.error(f"Malformed VAE state dictionary detected; missing required key '{ex}'. Reverting to default model {default_path}")
                pretrained_save_path = os.path.join(cache_dir, "models--{0}".format(
                    default_path.replace("/", "--")
                ))
                if not os.path.exists(pretrained_save_path) and task_callback is not None:
                    task_callback(f"Downloading default VAE weights from repository {default_path}")
                vae = AutoencoderKL.from_pretrained(default_path, cache_dir=cache_dir)
        elif os.path.exists(vae_path):
            if model_type in ["SDXL", "SDXL-Refiner"]:
                vae_config_path = os.path.join(cache_dir, "sdxl-vae-config.json")
                check_download(
                    "https://huggingface.co/stabilityai/sdxl-vae/raw/main/config.json",
                    vae_config_path,
                    check_size=False
                )
                vae = AutoencoderKL.from_config(
                    AutoencoderKL._dict_from_json_file(vae_config_path)
                )
                vae.load_state_dict(load_state_dict(vae_path), strict=False)
            else:
                vae = AutoencoderKL.from_single_file(
                    vae_path,
                    cache_dir=cache_dir,
                    from_safetensors = "safetensors" in vae_path
                )
        else:
            vae = AutoencoderKL.from_pretrained(vae_path, cache_dir=cache_dir)

        if vae_preview_path is None:
            if model_type in ["SDXL", "SDXL-Refiner"]:
                vae_preview_path = "madebyollin/taesdxl"
            else:
                vae_preview_path = "madebyollin/taesd"

        vae_preview_local_path = os.path.join(cache_dir, "models--{0}".format(vae_preview_path.replace("/", "--")))
        if not os.path.exists(vae_preview_local_path) and task_callback is not None:
            task_callback("Downloading preview VAE weights from repository {vae_preview_path}")
        vae_preview = AutoencoderTiny.from_pretrained(vae_preview_path, cache_dir=cache_dir)

        if offload_models:
            vae.to("cpu")
            empty_cache()

        if load_safety_checker:
            safety_checker_path = "CompVis/stable-diffusion-safety-checker"
            safety_checker_local_path = os.path.join(cache_dir, "models--{0}".format(safety_checker_path.replace("/", "--")))
            if not os.path.exists(safety_checker_local_path) and task_callback is not None:
                task_callback(f"Downloading safety checker weights from repository {safety_checker_path}")

            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                safety_checker_path,
                cache_dir=cache_dir
            )
            if offload_models:
                safety_checker.to("cpu")
                empty_cache()
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                safety_checker_path,
                cache_dir=cache_dir
            )
        else:
            safety_checker = None
            feature_extractor = None

        # Convert the text model.
        if model_type == "FrozenCLIPEmbedder":
            text_model = convert_ldm_clip_checkpoint(checkpoint)
            if offload_models:
                text_model.to("cpu")
                empty_cache()

            tokenizer_path = "openai/clip-vit-large-patch14"
            tokenizer_local_path = os.path.join(cache_dir, "models--{0}".format(tokenizer_path.replace("/", "--")))
            if not os.path.exists(tokenizer_local_path) and task_callback is not None:
                task_callback(f"Downloading tokenizer weights from repository {tokenizer_path}")

            tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=cache_dir
            )

            kwargs["text_encoder_2"] = None
            kwargs["tokenizer_2"] = None
            pipe = cls(
                vae=vae,
                vae_preview=vae_preview,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
                **kwargs,
            )
        elif model_type == "SDXL":
            tokenizer_path = "openai/clip-vit-large-patch14"
            tokenizer_local_path = os.path.join(cache_dir, "models--{0}".format(tokenizer_path.replace("/", "--")))
            if not os.path.exists(tokenizer_local_path) and task_callback is not None:
                task_callback(f"Downloading tokenizer weights from repository {tokenizer_path}")

            tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=cache_dir
            )
            text_encoder = convert_ldm_clip_checkpoint(checkpoint)

            tokenizer_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            tokenizer_2_local_path = os.path.join(cache_dir, "models--{0}".format(tokenizer_2_path.replace("/", "--")))
            if not os.path.exists(tokenizer_local_path) and task_callback is not None:
                task_callback(f"Downloading tokenizer 2 weights from repository {tokenizer_2_path}")

            tokenizer_2 = CLIPTokenizer.from_pretrained(
                tokenizer_2_path,
                cache_dir=cache_dir,
                pad_token="!"
            )
            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                tokenizer_2_path,
                prefix="conditioner.embedders.1.model.",
                has_projection=True,
                projection_dim=1280,
            )

            if offload_models:
                text_encoder.to("cpu")
                text_encoder_2.to("cpu")
                empty_cache()

            pipe = cls(
                vae=vae,
                vae_preview=vae_preview,
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
            tokenizer_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            tokenizer_2_local_path = os.path.join(cache_dir, "models--{0}".format(tokenizer_2_path.replace("/", "--")))
            if not os.path.exists(tokenizer_2_local_path) and task_callback is not None:
                task_callback(f"Downloading tokenizer 2 weights from repository {tokenizer_2_path}")

            tokenizer_2 = CLIPTokenizer.from_pretrained(
                tokenizer_2_path,
                cache_dir=cache_dir,
                pad_token="!"
            )
            text_encoder_2 = convert_open_clip_checkpoint(
                checkpoint,
                tokenizer_2_path,
                prefix="conditioner.embedders.0.model.",
                has_projection=True,
                projection_dim=1280,
            )

            if offload_models:
                text_encoder_2.to("cpu")
                empty_cache()

            pipe = cls(
                vae=vae,
                vae_preview=vae_preview,
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
            return pipe.to(torch_dtype=torch_dtype) # type: ignore[attr-defined]
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
        module_names, _ = self._get_signature_keys(self) # type: ignore[attr-defined]
        for name in module_names:
            module = getattr(self, name, None)
            if isinstance(module, torch.nn.Module):
                modules.append(module)
        modules.sort(key = lambda item: self.get_size_from_module(item), reverse=True)
        return modules

    def align_unet(
        self,
        device: torch.device,
        dtype: torch.dtype,
        freeu_factors: Optional[Tuple[float, float, float, float]] = None,
        offload_models: bool = False
    ) -> None:
        """
        Makes sure the unet is on the device and text encoders are off.
        """
        if offload_models:
            if self.text_encoder:
                self.text_encoder.to("cpu")
            if self.text_encoder_2:
                self.text_encoder_2.to("cpu")
            empty_cache()
        if freeu_factors is None:
            if getattr(self, "_freeu_enabled", False): # Make sure we've enabled this at least once
                self.unet.disable_freeu()
        else:
            s1, s2, b1, b2 = freeu_factors
            self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)
            self._freeu_enabled = True
        self.unet.to(device=device, dtype=dtype)

    def run_safety_checker(
        self,
        output: np.ndarray,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[np.ndarray, List[bool]]:
        """
        Override parent run_safety_checker to make sure safety checker is aligned
        """
        if self.safety_checker is not None:
            self.safety_checker.to(device)
        return super(EnfugueStableDiffusionPipeline, self).run_safety_checker(output, device, dtype) # type: ignore[misc]

    def load_ip_adapter(
        self,
        device: Union[str, torch.device],
        scale: float = 1.0,
        use_fine_grained: bool = False,
        use_face_model: bool = False,
        keepalive_callback: Optional[Callable[[], None]] = None
    ) -> None:
        """
        Loads the IP Adapter
        """
        if getattr(self, "ip_adapter", None) is None:
            raise RuntimeError("Pipeline does not have an IP adapter")
        if self.ip_adapter_loaded:
            altered = self.ip_adapter.set_scale( # type: ignore[union-attr]
                unet=self.unet,
                new_scale=scale,
                use_fine_grained=use_fine_grained,
                use_face_model=use_face_model,
                keepalive_callback=keepalive_callback,
                is_sdxl=self.is_sdxl,
                controlnets=self.controlnets
            )
            if altered == 0:
                logger.error("IP adapter appeared loaded, but setting scale did not modify it.")
                self.ip_adapter.load( # type: ignore[union-attr]
                    unet=self.unet,
                    is_sdxl=self.is_sdxl,
                    scale=scale,
                    use_fined_grained=use_fine_grained,
                    use_face_model=use_face_model,
                    keepalive_callback=keepalive_callback,
                    controlnets=self.controlnets
                )
        else:
            logger.debug("Loading IP adapter")
            self.ip_adapter.load( # type: ignore[union-attr]
                self.unet,
                is_sdxl=self.is_sdxl,
                scale=scale,
                keepalive_callback=keepalive_callback,
                use_fine_grained=use_fine_grained,
                use_face_model=use_face_model,
                controlnets=self.controlnets
            )
            self.ip_adapter_loaded = True

    def unload_ip_adapter(self) -> None:
        """
        Unloads the IP adapter by resetting attention processors to previous values
        """
        if getattr(self, "ip_adapter", None) is None:
            raise RuntimeError("Pipeline does not have an IP adapter")
        if self.ip_adapter_loaded:
            logger.debug("Unloading IP adapter")
            self.ip_adapter.unload(self.unet, self.controlnets) # type: ignore[union-attr]
            self.ip_adapter_loaded = False

    def get_image_embeds(
        self,
        image: PIL.Image.Image,
        num_images_per_prompt: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uses the IP adapter to get prompt embeddings from the image
        """
        if getattr(self, "ip_adapter", None) is None:
            raise RuntimeError("Pipeline does not have an IP adapter")
        image_prompt_embeds, image_uncond_prompt_embeds = self.ip_adapter.probe(image) # type: ignore[union-attr]
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        image_uncond_prompt_embeds = image_uncond_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        image_uncond_prompt_embeds = image_uncond_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return image_prompt_embeds, image_uncond_prompt_embeds

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
        clip_skip: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encodes the prompt into text encoder hidden states.
        See https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin): # type: ignore[unreachable]
            self._lora_scale = lora_scale # type: ignore[unreachable]

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

        # Align device
        if self.text_encoder:
            self.text_encoder = self.text_encoder.to(device, dtype=torch.float16)
        if self.text_encoder_2:
            self.text_encoder_2 = self.text_encoder_2.to(device, dtype=torch.float16)
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        if prompt_embeds is None:
            prompt_embeds_list = []
            for tokenizer, text_encoder, prompt in zip(tokenizers, text_encoders, prompts):
                if tokenizer is None or text_encoder is None:
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

                # Align device and encode
                text_input_ids = text_input_ids.to(device=device)
                text_encoder.to(device=device)
                prompt_embeds = text_encoder(
                    text_input_ids,
                    output_hidden_states=self.is_sdxl or clip_skip is not None,
                    attention_mask=attention_mask
                )

                if self.is_sdxl:
                    pooled_prompt_embeds = prompt_embeds[0]  # type: ignore
                    if clip_skip is None:
                        prompt_embeds = prompt_embeds.hidden_states[-2]  # type: ignore
                    else:
                        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)] # type: ignore
                elif clip_skip is not None:
                    prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)] # type: ignore
                    prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)
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
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt # type: ignore[attr-defined]
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
    def get_runtime_context(
        self,
        batch_size: int,
        device: Union[str, torch.device],
        ip_adapter_scale: Optional[Union[List[float], float]] = None,
        ip_adapter_plus: bool = False,
        ip_adapter_face: bool = False,
        step_complete: Optional[Callable[[bool], None]] = None
    ) -> Iterator[None]:
        """
        Builds the runtime context, which ensures everything is on the right devices
        """
        if isinstance(device, str):
            device = torch.device(device)
        if ip_adapter_scale is not None:
            self.load_ip_adapter(
                device=device,
                scale=max(ip_adapter_scale) if isinstance(ip_adapter_scale, list) else ip_adapter_scale,
                use_fine_grained=ip_adapter_plus,
                use_face_model=ip_adapter_face,
                keepalive_callback=None if step_complete is None else lambda: step_complete(False) # type: ignore[misc]
            )
        else:
            self.unload_ip_adapter()
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(device)

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
            state_dict = state_dict["state_dict"] # type: ignore[assignment]

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
        try:
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
            return super(EnfugueStableDiffusionPipeline, self).load_lora_weights( # type: ignore[misc]
                pretrained_model_name_or_path_or_dict, **kwargs
            )
        except (AttributeError, KeyError) as ex:
            if self.is_sdxl:
                message = "Are you trying to use a Stable Diffusion 1.5 LoRA with this Stable Diffusion XL pipeline?"
            else:
                message = "Are you trying to use a Stable Diffusion XL LoRA with this Stable Diffusion 1.5 pipeline?"
            raise IOError(f"Received {type(ex).__name__} loading LoRA. {message}")

    def load_sdxl_lora_weights(
        self, 
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        multiplier: float = 1.0,
        **kwargs: Any
    ) -> None:
        """
        Fix adapted from https://github.com/huggingface/diffusers/blob/4a4cdd6b07a36bbf58643e96c9a16d3851ca5bc5/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        """
        state_dict, network_alphas = self.lora_state_dict( # type: ignore[attr-defined]
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet) # type: ignore[attr-defined]

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder( # type: ignore[attr-defined]
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=multiplier,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder( # type: ignore[attr-defined]
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
        state_dict = safetensors.torch.load_file(pretrained_model_name_or_path_or_dict, device="cpu") # type: ignore[arg-type]

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

    def load_textual_inversion(self, inversion_path: str, **kwargs: Any) -> None:
        """
        Loads textual inversion
        Temporary implementation from https://github.com/huggingface/diffusers/issues/4405
        """
        try:
            if not self.is_sdxl:
                return super(EnfugueStableDiffusionPipeline, self).load_textual_inversion(inversion_path, **kwargs) # type: ignore[misc]

            logger.debug(f"Using SDXL adaptation for textual inversion - Loading {inversion_path}")
            if inversion_path.endswith("safetensors"):
                from safetensors import safe_open

                inversion = {}
                with safe_open(inversion_path, framework="pt", device="cpu") as f: # type: ignore[attr-defined]
                    for key in f.keys():
                        inversion[key] = f.get_tensor(key)
            else:
                inversion = torch.load(inversion_path, map_location="cpu")

            for i, (embedding_1, embedding_2) in enumerate(zip(inversion["clip_l"], inversion["clip_g"])):
                token = f"sksd{chr(i+65)}"
                if self.tokenizer is not None:
                    self.tokenizer.add_tokens(token)
                if self.tokenizer_2 is not None:
                    self.tokenizer_2.add_tokens(token)
                if self.text_encoder is not None and self.tokenizer is not None:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                    self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding_1
                if self.text_encoder_2 is not None:
                    if self.tokenizer_2 is not None:
                        token_id_2 = self.tokenizer_2.convert_tokens_to_ids(token)
                    elif self.tokenizer is not None:
                        token_id_2 = self.tokenizer.convert_tokens_to_ids(token)
                    else:
                        raise ValueError("No tokenizer, cannot add inversion to encoder")
                    self.text_encoder_2.resize_token_embeddings(len(self.tokenizer))
                    self.text_encoder_2.get_input_embeddings().weight.data[token_id_2] = embedding_2
        except (AttributeError, KeyError) as ex:
            if self.is_sdxl:
                message = "Are you trying to use a Stable Diffusion 1.5 textual inversion with this Stable Diffusion XL pipeline?"
            else:
                message = "Are you trying to use a Stable Diffusion XL textual inversion with this Stable Diffusion 1.5 pipeline?"
            raise IOError(f"Received {type(ex).__name__} loading textual inversion. {message}")

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
        random_latents = randn_tensor(
            shape,
            generator=generator,
            device=torch.device(device) if isinstance(device, str) else device,
            dtype=dtype
        )

        return random_latents * self.scheduler.init_noise_sigma # type: ignore[attr-defined]

    def encode_image_unchunked(
        self,
        image: torch.Tensor,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Encodes an image without chunking using the VAE.
        """
        logger.debug("Encoding image (unchunked).")
        if self.config.force_full_precision_vae: # type: ignore[attr-defined]
            self.vae.to(dtype=torch.float32)
            image = image.float()
        latents = self.vae.encode(image).latent_dist.sample(generator) * self.vae.config.scaling_factor # type: ignore[attr-defined]
        if self.config.force_full_precision_vae: # type: ignore[attr-defined]
            self.vae.to(dtype=dtype)
        return latents.to(dtype=dtype)

    def encode_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        weight_builder: MaskWeightBuilder,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[bool], None]] = None,
    ) -> torch.Tensor:
        """
        Encodes an image in chunks using the VAE.
        """
        _, _, height, width = image.shape
        chunks = self.get_chunks(height, width)
        total_steps = len(chunks)

        # Align device
        self.vae.to(device)

        if total_steps == 1:
            result = self.encode_image_unchunked(image, dtype, generator)
            if progress_callback is not None:
                progress_callback(True)
            return result

        logger.debug(f"Encoding image in {total_steps} steps.")

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        engine_latent_size = self.engine_size // self.vae_scale_factor
        num_channels = self.vae.config.latent_channels # type: ignore[attr-defined]

        count = torch.zeros((1, num_channels, latent_height, latent_width)).to(device=device)
        value = torch.zeros_like(count)

        if self.config.force_full_precision_vae: # type: ignore[attr-defined]
            self.vae.to(dtype=torch.float32)
            weight_builder.dtype = torch.float32
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

            # Build weights
            multiplier = weight_builder(
                mask_type=self.chunking_mask_type,
                batch=1,
                dim=num_channels,
                width=engine_latent_size,
                height=engine_latent_size,
                unfeather_left=left==0,
                unfeather_top=top==0,
                unfeather_right=right==latent_width,
                unfeather_bottom=bottom==latent_height,
                **self.chunking_mask_kwargs
            )

            value[:, :, top:bottom, left:right] += encoded_image * multiplier
            count[:, :, top:bottom, left:right] += multiplier

            if progress_callback is not None:
                progress_callback(True)

        if self.config.force_full_precision_vae: # type: ignore[attr-defined]
            self.vae.to(dtype=dtype)
            weight_builder.dtype = dtype
        return (torch.where(count > 0, value / count, value) * self.vae.config.scaling_factor).to(dtype=dtype) # type: ignore[attr-defined]

    def prepare_image_latents(
        self,
        image: Union[torch.Tensor, PIL.Image.Image],
        timestep: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        weight_builder: MaskWeightBuilder,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[bool], None]] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Prepares latents from an image, adding initial noise for img2img inference
        """
        image = image.to(device=device, dtype=dtype)
        if image.shape[1] == 4:
            init_latents = image
        else:
            init_latents = self.encode_image(
                image,
                device=device,
                generator=generator,
                dtype=dtype,
                weight_builder=weight_builder,
                progress_callback=progress_callback
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

        # add noise in accordance with timesteps
        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(
                shape,
                generator=generator,
                device=torch.device(device) if isinstance(device, str) else device,
                dtype=dtype
            )
            return self.scheduler.add_noise(init_latents, noise, timestep) # type: ignore[attr-defined]
        else:
            logger.debug("Not adding noise; starting from noised image.")
            return init_latents

    def prepare_mask_latents(
        self,
        mask: Union[PIL.Image.Image, torch.Tensor],
        image: Union[PIL.Image.Image, torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        weight_builder: MaskWeightBuilder,
        generator: Optional[torch.Generator] = None,
        do_classifier_free_guidance: bool = False,
        progress_callback: Optional[Callable[[bool], None]] = None,
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
            image,
            device=device,
            generator=generator,
            dtype=dtype,
            weight_builder=weight_builder,
            progress_callback=progress_callback,
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
        self,
        num_inference_steps: int,
        strength: Optional[float],
        denoising_start: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Gets the original timesteps from the scheduler based on strength when doing img2img
        """
        if denoising_start is None:
            if strength is None:
                raise ValueError("You must include at least one of 'denoising_start' or 'strength'")
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :] # type: ignore[attr-defined]

        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps # type: ignore[attr-defined]
                    - (denoising_start * self.scheduler.config.num_train_timesteps) # type: ignore[attr-defined]
                )
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            return torch.tensor(timesteps), len(timesteps)
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
            and self.config.requires_aesthetic_score # type: ignore[attr-defined]
        ):
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = None

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim # type: ignore[attr-defined]
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features # type: ignore[union-attr]

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim # type: ignore[attr-defined]
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetic_score` with `pipe.register_to_config(requires_aesthetic_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim # type: ignore[attr-defined]
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
        device: Union[str, torch.device],
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
        controlnet_conds: Optional[Dict[str, List[Tuple[torch.Tensor, float]]]],
        added_cond_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Executes the controlnet
        """
        if not controlnet_conds or not self.controlnets:
            return None, None
        down_blocks, mid_block = None, None
        for name in controlnet_conds:
            if self.controlnets.get(name, None) is None:
                raise RuntimeError(f"Conditioning image requested ControlNet {name}, but it's not loaded.")
            for controlnet_cond, conditioning_scale in controlnet_conds[name]:
                down_samples, mid_sample = self.controlnets[name](
                    latents,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=conditioning_scale,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                if down_blocks is None or mid_block is None: # type: ignore[unreachable]
                    down_blocks, mid_block = down_samples, mid_sample
                else:
                    down_blocks = [ # type: ignore[unreachable]
                        previous_block + current_block
                        for previous_block, current_block in zip(down_blocks, down_samples)
                    ]
                    mid_block += mid_sample
        return down_blocks, mid_block

    def denoise_unchunked(
        self,
        height: int,
        width: int,
        device: Union[str, torch.device],
        num_inference_steps: int,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        weight_builder: MaskWeightBuilder,
        guidance_scale: float,
        do_classifier_free_guidance: bool = False,
        is_inpainting_unet: bool = False,
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        control_images: PreparedControlImageArgType = None,
        progress_callback: Optional[Callable[[bool], None]] = None,
        latent_callback: Optional[Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
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
        num_warmup_steps = num_steps - num_inference_steps * self.scheduler.order # type: ignore[attr-defined]
        
        noise = None
        if mask is not None and mask_image is not None and not is_inpainting_unet:
            noise = latents.detach().clone() / self.scheduler.init_noise_sigma # type: ignore[attr-defined]
            noise = noise.to(device=device)

        logger.debug(f"Denoising image in {num_steps} steps on {device} (unchunked)")

        steps_since_last_callback = 0
        for i, t in enumerate(timesteps):
            # store ratio for later
            denoising_ratio = i / num_steps

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # type: ignore[attr-defined]

            # Get controlnet input(s) if configured
            if control_images is not None:
                # Find which control image(s) to use
                controlnet_conds: Dict[str, List[Tuple[torch.Tensor, float]]] = {}
                for controlnet_name in control_images:
                    for (
                        control_image,
                        conditioning_scale,
                        conditioning_start,
                        conditioning_end
                    ) in control_images[controlnet_name]:
                        if (
                            (conditioning_start is None or conditioning_start <= denoising_ratio) and
                            (conditioning_end is None or denoising_ratio <= conditioning_end)
                        ):
                            if controlnet_name not in controlnet_conds:
                                controlnet_conds[controlnet_name] = []

                            controlnet_conds[controlnet_name].append((control_image, conditioning_scale))
                if not controlnet_conds:
                    down_block, mid_block = None, None
                else:
                    down_block, mid_block = self.get_controlnet_conditioning_blocks(
                        device=device,
                        latents=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_conds=controlnet_conds,
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
            latents = self.scheduler.step( # type: ignore[attr-defined]
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
                    init_latents = self.scheduler.add_noise( # type: ignore[attr-defined]
                        init_latents,
                        noise,
                        torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents + init_mask * latents

            if progress_callback is not None:
                progress_callback(True)

            # call the callback, if provided
            steps_since_last_callback += 1
            if (
                latent_callback is not None
                and latent_callback_steps is not None
                and steps_since_last_callback >= latent_callback_steps
                and (i == num_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)) # type: ignore[attr-defined]
            ):
                steps_since_last_callback = 0
                latent_callback_value = latents

                if latent_callback_type != "latent":
                    latent_callback_value = self.decode_latent_preview(
                        latent_callback_value,
                        weight_builder=weight_builder,
                        device=device,
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
        weight_builder: MaskWeightBuilder,
        guidance_scale: float,
        do_classifier_free_guidance: bool = False,
        is_inpainting_unet: bool = False,
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        control_images: PreparedControlImageArgType = None,
        progress_callback: Optional[Callable[[bool], None]] = None,
        latent_callback: Optional[Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
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
                weight_builder=weight_builder,
                guidance_scale=guidance_scale,
                is_inpainting_unet=is_inpainting_unet,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mask=mask,
                mask_image=mask_image,
                image=image,
                control_images=control_images,
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
        num_warmup_steps = num_steps - num_inference_steps * self.scheduler.order # type: ignore[attr-defined]

        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor
        engine_latent_size = self.engine_size // self.vae_scale_factor

        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        samples, num_channels, _, _ = latents.shape

        total_num_steps = num_steps * num_chunks
        logger.debug(
            f"Denoising image in {total_num_steps} steps on {device} ({num_inference_steps} inference steps * {num_chunks} chunks)"
        )

        noise = None
        if mask is not None and mask_image is not None and not is_inpainting_unet:
            noise = latents.detach().clone() / self.scheduler.init_noise_sigma # type: ignore[attr-defined]
            noise = noise.to(device=device)

        steps_since_last_callback = 0
        for i, t in enumerate(timesteps):
            # Calculate ratio for later
            denoising_ratio = i / num_steps

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
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # type: ignore[attr-defined]

                    # Get controlnet input(s) if configured
                    if control_images is not None:
                        # Find which control image(s) to use
                        controlnet_conds: Dict[str, List[Tuple[torch.Tensor, float]]] = {}
                        for controlnet_name in control_images:
                            for (
                                control_image,
                                conditioning_scale,
                                conditioning_start,
                                conditioning_end
                            ) in control_images[controlnet_name]:
                                if (
                                    (conditioning_start is None or conditioning_start <= denoising_ratio) and
                                    (conditioning_end is None or denoising_ratio <= conditioning_end)
                                ):
                                    if controlnet_name not in controlnet_conds:
                                        controlnet_conds[controlnet_name] = []

                                    controlnet_conds[controlnet_name].append((
                                        control_image[:, :, top_px:bottom_px, left_px:right_px],
                                        conditioning_scale
                                    ))

                        if not controlnet_conds:
                            down_block, mid_block = None, None
                        else:
                            down_block, mid_block = self.get_controlnet_conditioning_blocks(
                                device=device,
                                latents=latent_model_input,
                                timestep=t,
                                encoder_hidden_states=prompt_embeds,
                                controlnet_conds=controlnet_conds,
                                added_cond_kwargs=added_cond_kwargs,
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
                    denoised_latents = self.scheduler.step( # type: ignore[attr-defined]
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
                        init_latents = self.scheduler.add_noise( # type: ignore[attr-defined]
                            init_latents,
                            noise[:, :, top:bottom, left:right],
                            torch.tensor([noise_timestep])
                        )

                    denoised_latents = (1 - init_mask) * init_latents + init_mask * denoised_latents

                # Build weights
                multiplier = weight_builder(
                    mask_type=self.chunking_mask_type,
                    batch=samples,
                    dim=num_channels,
                    width=engine_latent_size,
                    height=engine_latent_size,
                    unfeather_left=left==0,
                    unfeather_top=top==0,
                    unfeather_right=right==latent_width,
                    unfeather_bottom=bottom==latent_height,
                    **self.chunking_mask_kwargs
                )

                value[:, :, top:bottom, left:right] += denoised_latents * multiplier
                count[:, :, top:bottom, left:right] += multiplier

                # Call the progress callback
                if progress_callback is not None:
                    progress_callback(True)

            # multidiffusion
            latents = torch.where(count > 0, value / count, value)

            # call the latent_callback, if provided
            steps_since_last_callback += 1
            if (
                latent_callback is not None
                and latent_callback_steps is not None
                and steps_since_last_callback >= latent_callback_steps
                and (i == num_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)) # type: ignore[attr-defined]
            ):
                steps_since_last_callback = 0
                latent_callback_value = latents

                if latent_callback_type != "latent":
                    latent_callback_value = self.decode_latent_preview(
                        latent_callback_value,
                        weight_builder=weight_builder,
                        device=device,
                    )
                    latent_callback_value = self.denormalize_latents(latent_callback_value)
                    if latent_callback_type != "pt":
                        latent_callback_value = self.image_processor.pt_to_numpy(latent_callback_value)
                        if latent_callback_type == "pil":
                            latent_callback_value = self.image_processor.numpy_to_pil(latent_callback_value)
                latent_callback(latent_callback_value)

        return latents

    def decode_latent_preview(
        self,
        latents: torch.Tensor,
        weight_builder: MaskWeightBuilder,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """
        Issues the command to decode latents with the tiny VAE.
        Batches anything > 1024px (128 latent)
        """
        from math import ceil
        batch = latents.shape[0]
        height, width = latents.shape[-2:]
        height_px = height * self.vae_scale_factor
        width_px = width * self.vae_scale_factor

        max_size = 128
        overlap = 16
        max_size_px = max_size * self.vae_scale_factor

        if height > max_size or width > max_size:
            # Do some chunking to avoid sharp lines, but don't follow global chunk
            width_chunks = ceil(width / (max_size - overlap))
            height_chunks = ceil(height / (max_size - overlap))
            decoded_preview = torch.zeros(
                (batch, 3, height*self.vae_scale_factor, width*self.vae_scale_factor),
                dtype=latents.dtype,
                device=device
            )
            multiplier = torch.zeros_like(decoded_preview)
            for i in range(height_chunks):
                start_h = max(0, i * (max_size - overlap))
                end_h = start_h + max_size
                if end_h > height:
                    diff = end_h - height
                    end_h -= diff
                    start_h = max(0, start_h-diff)
                start_h_px = start_h * self.vae_scale_factor
                end_h_px = end_h * self.vae_scale_factor
                for j in range(width_chunks):
                    start_w = max(0, j * (max_size - overlap))
                    end_w = start_w + max_size
                    if end_w > width:
                        diff = end_w - width
                        end_w -= diff
                        start_w = max(0, start_w-diff)
                    start_w_px = start_w * self.vae_scale_factor
                    end_w_px = end_w * self.vae_scale_factor
                    mask = weight_builder(
                        mask_type="bilinear",
                        batch=batch,
                        dim=3,
                        width=min(width_px, max_size_px),
                        height=min(height_px, max_size_px),
                        unfeather_left=start_w==0,
                        unfeather_top=start_h==0,
                        unfeather_right=end_w==width,
                        unfeather_bottom=end_h==height,
                    )
                    decoded_view = self.vae_preview.decode(
                        latents[:, :, start_h:end_h, start_w:end_w],
                        return_dict=False
                    )[0].to(device)
                    decoded_preview[:, :, start_h_px:end_h_px, start_w_px:end_w_px] += decoded_view * mask
                    multiplier[:, :, start_h_px:end_h_px, start_w_px:end_w_px] += mask
            return decoded_preview / multiplier
        else:
            return self.vae_preview.decode(latents, return_dict=False)[0].to(device)

    def decode_latent_view(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Issues the command to decode a chunk of latents with the VAE.
        """
        return self.vae.decode(latents, return_dict=False)[0]

    def decode_latents_unchunked(
        self,
        latents: torch.Tensor,
        device: Union[str, torch.device]
    ) -> torch.Tensor:
        """
        Decodes the latents using the VAE without chunking.
        """
        return self.decode_latent_view(latents).to(device=device)

    def decode_latents(
        self,
        latents: torch.Tensor,
        device: Union[str, torch.device],
        weight_builder: MaskWeightBuilder,
        progress_callback: Optional[Callable[[bool], None]] = None,
    ) -> torch.Tensor:
        """
        Decodes the latents in chunks as necessary.
        """
        samples, num_channels, height, width = latents.shape
        height *= self.vae_scale_factor
        width *= self.vae_scale_factor

        latents = 1 / self.vae.config.scaling_factor * latents # type: ignore[attr-defined]

        chunks = self.get_chunks(height, width)
        total_steps = len(chunks)

        revert_dtype = None

        if self.config.force_full_precision_vae: # type: ignore[attr-defined]
            # Resist overflow
            revert_dtype = latents.dtype
            self.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)
            weight_builder.dtype = torch.float32

        if total_steps <= 1:
            result = self.decode_latents_unchunked(latents, device)
            if progress_callback is not None:
                progress_callback(True)
            if self.config.force_full_precision_vae: # type: ignore[attr-defined]
                self.vae.to(dtype=latents.dtype)
            return result

        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor
        engine_latent_size = self.engine_size // self.vae_scale_factor

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

            # Build weights
            multiplier = weight_builder(
                mask_type=self.chunking_mask_type,
                batch=samples,
                dim=3,
                width=self.engine_size,
                height=self.engine_size,
                unfeather_left=left==0,
                unfeather_top=top==0,
                unfeather_right=right==latent_width,
                unfeather_bottom=bottom==latent_height,
                **self.chunking_mask_kwargs
            )

            value[:, :, top_px:bottom_px, left_px:right_px] += decoded_latents * multiplier
            count[:, :, top_px:bottom_px, left_px:right_px] += multiplier

            if progress_callback is not None:
                progress_callback(True)

        # re-average pixels
        latents = torch.where(count > 0, value / count, value)
        if revert_dtype is not None:
            latents = latents.to(dtype=revert_dtype)
            self.vae.to(dtype=revert_dtype)
        return latents

    def prepare_extra_step_kwargs(
        self,
        generator: Optional[torch.Generator],
        eta: float
    ) -> Dict[str, Any]:
        """
        Prepares arguments to add during denoising
        """
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()) # type: ignore[attr-defined]
        extra_step_kwargs: Dict[str, Any] = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()) # type: ignore[attr-defined]
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_step_complete_callback(
        self,
        overall_steps: int,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        log_interval: int = 10,
        log_sampling_duration: Union[int, float] = 5,
    ) -> Callable[[bool], None]:
        """
        Creates a scoped callback to trigger during iterations
        """
        overall_step, window_start_step, its = 0, 0, 0.0
        window_start = datetime.datetime.now()
        digits = math.ceil(math.log10(overall_steps))

        def step_complete(increment_step: bool = True) -> None:
            nonlocal overall_step, window_start, window_start_step, its
            if increment_step:
                overall_step += 1
            if overall_step != 0 and overall_step % log_interval == 0 or overall_step == overall_steps:
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
        device: Optional[Union[str, torch.device]] = None,
        offload_models: bool = False,
        prompt: Optional[str] = None,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        clip_skip: Optional[int] = None,
        freeu_factors: Optional[Tuple[float, float, float, float]] = None,
        image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
        mask: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
        control_images: ControlImageArgType = None,
        ip_adapter_images: ImagePromptArgType = None,
        ip_adapter_plus: bool = False,
        ip_adapter_face: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        chunking_size: Optional[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        strength: Optional[float] = 0.8,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        noise_generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Literal["latent", "pt", "np", "pil"] = "pil",
        return_dict: bool = True,
        scale_image: bool = True,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        latent_callback: Optional[Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]] = None,
        latent_callback_steps: Optional[int] = None,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        chunking_mask_type: Optional[MASK_TYPE_LITERAL] = None,
        chunking_mask_kwargs: Optional[Dict[str, Any]] = None,
        noise_offset: Optional[float] = None,
        noise_method: NOISE_METHOD_LITERAL = "perlin",
        noise_blend_method: LATENT_BLEND_METHOD_LITERAL = "inject",
    ) -> Union[
        StableDiffusionPipelineOutput,
        Tuple[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]], Optional[List[bool]]],
    ]:
        """
        Invokes the pipeline.
        """
        # 1. Default height and width to image or unet config
        if not height:
            if image is not None:
                if isinstance(image, str):
                    image = PIL.Image.open(image)
                if isinstance(image, PIL.Image.Image):
                    _, height = image.size
                else:
                    height = image.shape[-2] * self.vae_scale_factor
            else:
                height = self.unet.config.sample_size * self.vae_scale_factor # type: ignore[attr-defined]

        if not width:
            if image is not None:
                if isinstance(image, str):
                    image = PIL.Image.open(image)
                if isinstance(image, PIL.Image.Image):
                    width, _ = image.size
                else:
                    width = image.shape[-1] * self.vae_scale_factor
            else:
                width = self.unet.config.sample_size * self.vae_scale_factor # type: ignore[attr-defined]

        height = cast(int, height)
        width = cast(int, width)

        # Training details
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # Allow overridding chunking variables
        if chunking_size is not None:
            self.chunking_size = chunking_size
        if chunking_mask_type is not None:
            self.chunking_mask_type = chunking_mask_type
        if chunking_mask_kwargs is not None:
            self.chunking_mask_kwargs = chunking_mask_kwargs
        
        # Check latent callback steps, disable if 0 or maximum offloading set
        if latent_callback_steps == 0:
            latent_callback_steps = None

        # Standardize IP adapter tuples
        ip_adapter_tuples: Optional[List[Tuple[PIL.Image.Image, float]]] = None
        if ip_adapter_images:
            ip_adapter_tuples = []
            for ip_adapter_argument in (ip_adapter_images if isinstance(ip_adapter_images, list) else [ip_adapter_images]):
                if isinstance(ip_adapter_argument, dict):
                    ip_adapter_tuples.append((
                        ip_adapter_argument["image"],
                        ip_adapter_argument.get("scale", 1.0)
                    ))
                elif isinstance(ip_adapter_argument, list):
                    ip_adapter_tuples.append(tuple(ip_adapter_argument)) # type: ignore[arg-type]
                elif isinstance(ip_adapter_argument, tuple):
                    ip_adapter_tuples.append(ip_adapter_argument) # type: ignore[arg-type]
                else:
                    ip_adapter_tuples.append((ip_adapter_argument, 1.0))
            ip_adapter_scale = max([scale for img, scale in ip_adapter_tuples])
        else:
            ip_adapter_tuples = None
            ip_adapter_scale = None

        # Convenient bool for later
        decode_intermediates = latent_callback_steps is not None and latent_callback is not None

        # Define outputs here to process later
        prepared_latents: Optional[torch.Tensor] = None
        output_nsfw: Optional[List[bool]] = None

        # Determine dimensionality
        is_inpainting_unet = self.unet.config.in_channels == 9 # type: ignore[attr-defined]

        # Define call parameters
        if prompt is not None:
            batch_size = 1
        elif prompt_embeds:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Prompt or prompt embeds are required.")

        if is_inpainting_unet:
            if image is None:
                logger.warning("No image present, but using inpainting model. Adding blank image.")
                image = PIL.Image.new("RGB", (width, height))
            if mask is None:
                logger.warning("No mask present, but using inpainting model. Adding blank mask.")
                mask = PIL.Image.new("RGB", (width, height), (255, 255, 255))

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        do_classifier_free_guidance = guidance_scale > 1.0

        # Calculate chunks
        num_chunks = max(1, len(self.get_chunks(height, width)))
        self.scheduler.set_timesteps(num_inference_steps, device=device) # type: ignore[attr-defined]

        if image is not None and mask is None and (strength is not None or denoising_start is not None):
            # Scale timesteps by strength
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps,
                strength,
                denoising_start=denoising_start
            )
        else:
            timesteps = self.scheduler.timesteps # type: ignore[attr-defined]

        batch_size *= num_images_per_prompt
        num_scheduled_inference_steps = len(timesteps)

        # Calculate end of steps if we aren't going all the way
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps # type: ignore[attr-defined]
                    - (denoising_end * self.scheduler.config.num_train_timesteps) # type: ignore[attr-defined]
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
            num_scheduled_inference_steps = len(timesteps)

        # Calculate total steps including all unet and vae calls
        encoding_steps = 0
        decoding_steps = 1

        if image is not None and type(image) is not torch.Tensor:
            encoding_steps += 1
        if mask is not None and type(mask) is not torch.Tensor:
            encoding_steps += 1
            if not is_inpainting_unet:
                encoding_steps += 1

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
                
        if self.config.force_full_precision_vae: # type: ignore[attr-defined]
            logger.debug(f"Configuration indicates VAE must be used in full precision")
            # make sure the VAE is in float32 mode, as it overflows in float16
            self.vae.to(dtype=torch.float32)
        elif self.is_sdxl:
            logger.debug(f"Configuration indicates VAE may operate in half precision")
            self.vae.to(dtype=torch.float16)

        with self.get_runtime_context(
            batch_size=batch_size,
            device=device,
            ip_adapter_scale=ip_adapter_scale,
            ip_adapter_plus=ip_adapter_plus,
            ip_adapter_face=ip_adapter_face,
            step_complete=step_complete
        ):
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
                    negative_prompt_2=negative_prompt_2,
                    clip_skip=clip_skip
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
                    negative_prompt_2=negative_prompt_2,
                    clip_skip=clip_skip
                )  # type: ignore
                pooled_prompt_embeds = None
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Open images if they're files
            if isinstance(image, str):
                image = PIL.Image.open(image)

            if isinstance(mask, str):
                mask = PIL.Image.open(mask)

            # Scale images if requested
            if scale_image and isinstance(image, PIL.Image.Image):
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
            if isinstance(image, PIL.Image.Image):
                image = image.convert("RGB")
            if isinstance(mask, PIL.Image.Image):
                mask = mask.convert("L")
            if isinstance(image, PIL.Image.Image) and isinstance(mask, PIL.Image.Image):
                if is_inpainting_unet:
                    prepared_mask, prepared_image = self.prepare_mask_and_image(mask, image, False) # type: ignore
                    init_image = None
                else:
                    prepared_mask, prepared_image, init_image = self.prepare_mask_and_image(mask, image, True) # type: ignore
            elif image is not None and mask is None:
                prepared_image = self.image_processor.preprocess(image)
                prepared_mask, prepared_image_latents, init_image = None, None, None
            else:
                prepared_mask, prepared_image, prepared_image_latents, init_image = None, None, None, None

            if width < self.engine_size or height < self.engine_size:
                # Disable chunking
                logger.debug(f"{width}x{height} is smaller than is chunkable, disabling.")
                self.chunking_size = 0

            # No longer none
            prompt_embeds = cast(torch.Tensor, prompt_embeds)

            # Build the weight builder
            weight_builder = MaskWeightBuilder(device=device, dtype=prompt_embeds.dtype)
            with weight_builder:
                if prepared_image is not None and prepared_mask is not None:
                    # Inpainting
                    num_channels_latents = self.vae.config.latent_channels # type: ignore[attr-defined]

                    if latents:
                        prepared_latents = latents.to(device) * self.scheduler.init_noise_sigma # type: ignore[attr-defined]
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
                        mask=prepared_mask,
                        image=prepared_image,
                        batch_size=batch_size,
                        height=height,
                        width=width,
                        dtype=prompt_embeds.dtype,
                        device=device,
                        weight_builder=weight_builder,
                        generator=generator,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        progress_callback=step_complete,
                    )

                    if init_image is not None:
                        init_image = init_image.to(device=device, dtype=prompt_embeds.dtype)
                        init_image = self.encode_image(
                            init_image,
                            device=device,
                            dtype=prompt_embeds.dtype,
                            generator=generator,
                            weight_builder=weight_builder
                        )

                    # prepared_latents = noise or init latents + noise
                    # prepared_mask = only mask
                    # prepared_image_latents = masked image
                    # init_image = only image when not using inpainting unet
                elif prepared_image is not None and strength is not None:
                    # img2img
                    prepared_latents = self.prepare_image_latents(
                        image=prepared_image,
                        timestep=timesteps[:1].repeat(batch_size),
                        batch_size=batch_size,
                        dtype=prompt_embeds.dtype,
                        device=device,
                        weight_builder=weight_builder,
                        generator=generator,
                        progress_callback=step_complete,
                        add_noise=denoising_start is None,
                    )
                    # prepared_latents = img + noise
                elif latents:
                    prepared_latents = latents.to(device) * self.scheduler.init_noise_sigma # type: ignore[attr-defined]
                    # prepared_latents = passed latents + noise
                else:
                    # txt2img
                    prepared_latents = self.create_latents(
                        batch_size=batch_size,
                        num_channels_latents=self.unet.config.in_channels, # type: ignore[attr-defined]
                        height=height,
                        width=width,
                        dtype=prompt_embeds.dtype,
                        device=device,
                        generator=generator,
                    )
                    # prepared_latents = noise

                # Look for controlnet and conditioning image
                prepared_control_images: PreparedControlImageArgType = {}
                if control_images is not None:
                    if not self.controlnets:
                        logger.warning("Control image passed, but no controlnet present. Ignoring.")
                        prepared_control_images = None
                    else:
                        for name in control_images:
                            if name not in self.controlnets:
                                raise RuntimeError(f"Control image mapped to ControlNet {name}, but it is not loaded.")

                            image_list = control_images[name]
                            if not isinstance(image_list, list):
                                image_list = [image_list]

                            for controlnet_image in image_list:
                                if isinstance(controlnet_image, tuple):
                                    if len(controlnet_image) == 4:
                                        controlnet_image, conditioning_scale, conditioning_start, conditioning_end = controlnet_image
                                    elif len(controlnet_image) == 3:
                                        controlnet_image, conditioning_scale, conditioning_start = controlnet_image
                                        conditioning_end = None
                                    elif len(controlnet_image) == 2:
                                        controlnet_image, conditioning_scale = controlnet_image
                                        conditioning_start, conditioning_end = None, None

                                elif isinstance(controlnet_image, dict):
                                    conditioning_scale = controlnet_image.get("scale", 1.0)
                                    conditioning_start = controlnet_image.get("start", None)
                                    conditioning_end = controlnet_image.get("end", None)
                                    controlnet_image = controlnet_image["image"]
                                else:
                                    conditioning_scale = 1.0
                                    conditioning_start, conditioning_end = None, None

                                if isinstance(controlnet_image, str):
                                    controlnet_image = PIL.Image.open(controlnet_image)

                                prepared_controlnet_image = self.prepare_control_image(
                                    image=controlnet_image,
                                    height=height,
                                    width=width,
                                    batch_size=batch_size,
                                    num_images_per_prompt=num_images_per_prompt,
                                    device=device,
                                    dtype=prompt_embeds.dtype,
                                    do_classifier_free_guidance=do_classifier_free_guidance,
                                )

                                if name not in prepared_control_images: # type: ignore[operator]
                                    prepared_control_images[name] = [] # type: ignore[index]

                                prepared_control_images[name].append( # type: ignore[index]
                                    (prepared_controlnet_image, conditioning_scale, conditioning_start, conditioning_end)
                                )

                # Should no longer be None
                prepared_latents = cast(torch.Tensor, prepared_latents)

                # Prepare extra step kwargs
                extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

                # Get prompt embeds here if using ip adapter
                if ip_adapter_tuples is not None:
                    image_prompt_embeds = torch.Tensor().to(
                        device=device,
                        dtype=prompt_embeds.dtype
                    )
                    image_uncond_prompt_embeds = torch.Tensor().to(
                        device=device,
                        dtype=prompt_embeds.dtype
                    )
                    for img, scale in ip_adapter_tuples:
                        these_prompt_embeds, these_uncond_prompt_embeds = self.get_image_embeds(
                            img,
                            num_images_per_prompt
                        )

                        image_prompt_embeds = torch.cat([
                            image_prompt_embeds,
                            (these_prompt_embeds * scale / ip_adapter_scale)
                        ], dim=1)

                        image_uncond_prompt_embeds = torch.cat([
                            image_uncond_prompt_embeds,
                            (these_uncond_prompt_embeds * scale / ip_adapter_scale)
                        ], dim=1)

                    if self.is_sdxl:
                        negative_prompt_embeds = cast(torch.Tensor, negative_prompt_embeds)
                        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                        negative_prompt_embeds = torch.cat([negative_prompt_embeds, image_uncond_prompt_embeds], dim=1)
                    else:
                        if do_classifier_free_guidance:
                            _negative_prompt_embeds, _prompt_embeds = prompt_embeds.chunk(2)
                        else:
                            _negative_prompt_embeds, _prompt_embeds = negative_prompt_embeds, prompt_embeds # type: ignore
                        prompt_embeds = torch.cat([_prompt_embeds, image_prompt_embeds], dim=1)
                        negative_prompt_embeds = torch.cat([_negative_prompt_embeds, image_uncond_prompt_embeds], dim=1)
                        if do_classifier_free_guidance:
                            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                    if ip_adapter_images:
                        del ip_adapter_images

                # Prepared added time IDs and embeddings (SDXL)
                if self.is_sdxl:
                    negative_prompt_embeds = cast(torch.Tensor, negative_prompt_embeds)
                    pooled_prompt_embeds = cast(torch.Tensor, pooled_prompt_embeds)
                    negative_pooled_prompt_embeds = cast(torch.Tensor, negative_pooled_prompt_embeds)
                    add_text_embeds = pooled_prompt_embeds

                    if do_classifier_free_guidance:
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)

                    if self.config.requires_aesthetic_score: # type: ignore[attr-defined]
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

                # Unload VAE, and maybe preview VAE
                self.vae.to("cpu")
                if decode_intermediates:
                    self.vae_preview.to(device)
                empty_cache()

                # Inject noise
                if noise_offset is not None and noise_offset > 0 and denoising_start is None:
                    noise_timestep = timesteps[:1].repeat(batch_size).to("cpu", dtype=torch.int)
                    schedule_factor = (1 - self.scheduler.alphas_cumprod[noise_timestep]) ** 0.5 # type: ignore
                    schedule_factor = schedule_factor.flatten()[0] # type: ignore
                    logger.debug(f"Adding {noise_method} noise by method {noise_blend_method} and factor {schedule_factor*noise_offset:.4f} - offset is {noise_offset:.2f}, scheduled alpha cumulative product is {schedule_factor:.4f}")
                    noise_latents = make_noise(
                        batch_size=prepared_latents.shape[0],
                        channels=prepared_latents.shape[1],
                        height=height // self.vae_scale_factor,
                        width=width // self.vae_scale_factor,
                        generator=noise_generator,
                        device=device,
                        dtype=prepared_latents.dtype,
                        method=noise_method,
                    )
                    prepared_latents = blend_latents(
                        left=prepared_latents,
                        right=noise_latents,
                        time=noise_offset,
                        method=noise_blend_method
                    )

                # Make sure controlnet on device
                if self.controlnets is not None:
                    for name in self.controlnets:
                        self.controlnets[name].to(device=device)

                # Make sure unet is on device
                self.align_unet(
                    device=device,
                    dtype=prompt_embeds.dtype,
                    freeu_factors=freeu_factors,
                    offload_models=offload_models
                ) # May be overridden by RT

                # Denoising loop
                prepared_latents = self.denoise(
                    height=height,
                    width=width,
                    device=device,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                    latents=prepared_latents,
                    prompt_embeds=prompt_embeds,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    is_inpainting_unet=is_inpainting_unet,
                    mask=prepared_mask,
                    mask_image=prepared_image_latents,
                    image=init_image,
                    control_images=prepared_control_images,
                    progress_callback=step_complete,
                    latent_callback=latent_callback,
                    latent_callback_steps=latent_callback_steps,
                    latent_callback_type=latent_callback_type,
                    extra_step_kwargs=extra_step_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    weight_builder=weight_builder
                )

                # Clear no longer needed tensors
                del prepared_mask
                del prepared_image_latents

                # Unload controlnets to free memory
                if self.controlnets is not None:
                    for name in self.controlnets:
                        self.controlnets[name].to("cpu")
                    del prepared_control_images

                # Unload UNet to free memory
                if offload_models:
                    self.unet.to("cpu")

                # Empty caches for more memory
                empty_cache()
                if output_type != "latent":
                    self.vae.to(
                        dtype=torch.float32 if self.config.force_full_precision_vae else prepared_latents.dtype, # type: ignore[attr-defined]
                        device=device
                    )
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

                    prepared_latents = self.decode_latents(
                        prepared_latents,
                        device=device,
                        progress_callback=step_complete,
                        weight_builder=weight_builder
                    )

                    if self.config.force_full_precision_vae: # type: ignore[attr-defined]
                        self.vae.to(dtype=prepared_latents.dtype)

            if output_type == "latent":
                output = prepared_latents
            else:
                if offload_models:
                    # Offload VAE again
                    self.vae.to("cpu")
                    self.vae_preview.to("cpu")
                    empty_cache()
                output = self.denormalize_latents(prepared_latents)
                if output_type != "pt":
                    output = self.image_processor.pt_to_numpy(output)
                    output_nsfw = self.run_safety_checker(output, device, prompt_embeds.dtype)[1]# type: ignore[arg-type]
                    if output_type == "pil":
                        output = self.image_processor.numpy_to_pil(output)

            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()

            empty_cache()

            if not return_dict:
                return (output, output_nsfw)

            return StableDiffusionPipelineOutput(images=output, nsfw_content_detected=output_nsfw) # type: ignore[arg-type]
