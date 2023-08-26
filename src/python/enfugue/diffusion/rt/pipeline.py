import torch
import tensorrt as trt

from typing import Optional, List, Dict, Iterator, Any, Union, Tuple
from typing_extensions import Self

from contextlib import contextmanager

from polygraphy import cuda
from transformers import (
    CLIPImageProcessor,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel

from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from enfugue.util import logger, TokenMerger
from enfugue.diffusion.pipeline import EnfugueStableDiffusionPipeline
from enfugue.diffusion.util import DTypeConverter
from enfugue.diffusion.rt.engine import Engine
from enfugue.diffusion.rt.model import BaseModel, UNet, VAE, CLIP, ControlledUNet, ControlNet


class EnfugueTensorRTStableDiffusionPipeline(EnfugueStableDiffusionPipeline):
    models: Dict[str, BaseModel]
    engine: Dict[str, Engine]

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
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetic_score: bool = False,
        force_full_precision_vae: bool = False,
        engine_size: int = 512,  # Recommended even for machines that can handle more
        chunking_size: int = 32,
        chunking_blur: int = 64,
        max_batch_size: int = 16,
        # ONNX export parameters
        force_engine_rebuild: bool = False,
        vae_engine_dir: Optional[str] = None,
        clip_engine_dir: Optional[str] = None,
        unet_engine_dir: Optional[str] = None,
        controlled_unet_engine_dir: Optional[str] = None,
        controlnet_engine_dir: Optional[str] = None,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = False,
        build_preview_features: bool = False,
        onnx_opset: int = 17,
    ) -> None:
        super(EnfugueTensorRTStableDiffusionPipeline, self).__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            force_full_precision_vae=force_full_precision_vae,
            requires_aesthetic_score=requires_aesthetic_score,
            engine_size=engine_size,
            chunking_size=chunking_size,
            chunking_blur=chunking_blur,
        )

        if self.controlnet is not None:
            # Hijack forward
            self.unet.forward = self.controlled_unet_forward
        self.vae.forward = self.vae.decode
        self.onnx_opset = onnx_opset
        self.force_engine_rebuild = force_engine_rebuild
        self.vae_engine_dir = vae_engine_dir
        self.clip_engine_dir = clip_engine_dir
        self.unet_engine_dir = unet_engine_dir
        self.controlled_unet_engine_dir = controlled_unet_engine_dir
        self.controlnet_engine_dir = controlnet_engine_dir
        self.build_static_batch = build_static_batch
        self.build_dynamic_shape = build_dynamic_shape
        self.build_preview_features = build_preview_features
        self.max_batch_size = max_batch_size

        if self.build_dynamic_shape or self.engine_size > 512:
            self.max_batch_size = 4

        # Set default to DDIM - The PNDM default that some models have does not work with TRT
        if not isinstance(self.scheduler, DDIMScheduler):
            logger.debug(f"TensorRT pipeline changing default scheduler from {type(self.scheduler).__name__} to DDIM")
            self.scheduler = DDIMScheduler.from_config(self.scheduler_config)

        self.stream = None  # loaded in load_resources()
        self.models = {}  # loaded in load_models()
        self.engine = {}  # loaded in build_engines()

    @staticmethod
    def device_view(t: torch.Tensor) -> cuda.DeviceView:
        """
        Gets a device view over a tensor
        """
        return cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=DTypeConverter.from_torch(t.dtype))

    def controlled_unet_forward(self, *args, **kwargs):
        """
        Highjacks unet forwarding.
        """
        sample, timestep, embeddings, mid_block = args[:4]
        down_blocks = args[4:]
        return UNet2DConditionModel.forward(
            self.unet,
            sample,
            timestep,
            embeddings,
            mid_block_additional_residual=mid_block,
            down_block_additional_residuals=down_blocks,
        )

    def load_models(self, use_fp16: bool = False) -> None:
        """
        Loads pipeline models to build later.
        """
        if not self.text_encoder:
            raise ValueError("Missing text encoder, cannot get embedding dimensions.")
        self.embedding_dim = self.text_encoder.config.hidden_size
        models_args = {
            "device": self.torch_device,
            "max_batch_size": self.max_batch_size,
            "embedding_dim": self.embedding_dim,
            "use_fp16": use_fp16,
        }
        if self.clip_engine_dir is not None:
            self.models["clip"] = CLIP(self.text_encoder, **models_args)
        if self.unet_engine_dir is not None:
            self.models["unet"] = UNet(self.unet, unet_dim=self.unet.config.in_channels, **models_args)
        if self.controlled_unet_engine_dir is not None:
            self.models["controlledunet"] = ControlledUNet(
                self.unet, unet_dim=self.unet.config.in_channels, **models_args
            )
        if self.controlnet_engine_dir is not None and self.controlnet is not None:
            self.models["controlnet"] = ControlNet(
                self.controlnet, unet_dim=self.unet.config.in_channels, **models_args
            )
        if self.vae_engine_dir is not None:
            self.models["vae"] = VAE(self.vae, **models_args)

    def load_resources(self, image_height: int, image_width: int, batch_size: int) -> None:
        """
        Allocates buffers for configured RT models.
        """
        self.stream = cuda.Stream()

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width),
                device=self.torch_device,
            )

    @contextmanager
    def get_runtime_context(self, batch_size: int, device: Union[str, torch.device]) -> Iterator[None]:
        """
        We initialize the TensorRT runtime here.
        """
        self.load_resources(self.engine_size, self.engine_size, batch_size)
        with (
            torch.inference_mode(),
            torch.autocast(device.type if isinstance(device, torch.device) else device),
            trt.Runtime(trt.Logger(trt.Logger.ERROR)),
        ):
            if self.vae is not None and self.vae_engine_dir is None:
                self.vae.to(device)
            if self.controlnet is not None and self.controlnet_engine_dir is None:
                self.controlnet.to(device)
            if self.text_encoder is not None and self.clip_engine_dir is None:
                self.text_encoder.to(device)
            if self.unet_engine_dir is not None or self.controlled_unet_engine_dir is not None:
                logger.info("TensorRT unloading UNET")
                self.unet = self.unet.to("cpu")
            yield
            if self.unet_engine_dir is not None or self.controlled_unet_engine_dir is not None:
                logger.info("TensorRT reloading UNET")
                self.unet = self.unet.to(device)

    def to(
        self,
        torch_device: Optional[Union[str, torch.device]] = None,
        silence_dtype_warnings: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ) -> Self:
        """
        First follow the parent to(), then load engines.
        """
        super().to(torch_device, silence_dtype_warnings=silence_dtype_warnings)

        if torch_device == "cpu" or isinstance(torch_device, torch.device) and torch_device.type == "cpu":
            return self
        elif getattr(self, "_built", False):
            return self

        # set device
        self.torch_device = self._execution_device

        # load models
        use_fp16 = torch_dtype is torch.float16
        self.load_models(use_fp16)

        # build engines
        engines_to_build: Dict[str, BaseModel] = {}
        if self.clip_engine_dir is not None:
            engines_to_build[self.clip_engine_dir] = self.models["clip"]
        if self.unet_engine_dir is not None:
            engines_to_build[self.unet_engine_dir] = self.models["unet"]
        if self.controlled_unet_engine_dir is not None:
            engines_to_build[self.controlled_unet_engine_dir] = self.models["controlledunet"]
        if self.controlnet_engine_dir is not None:
            engines_to_build[self.controlnet_engine_dir] = self.models["controlnet"]
        if self.vae_engine_dir is not None:
            engines_to_build[self.vae_engine_dir] = self.models["vae"]

        self.engine = Engine.build_all(
            engines_to_build,
            self.onnx_opset,
            use_fp16=use_fp16,
            opt_image_height=self.engine_size,
            opt_image_width=self.engine_size,
            force_engine_rebuild=self.force_engine_rebuild,
            static_batch=self.build_static_batch,
            static_shape=not self.build_dynamic_shape,
            enable_preview=self.build_preview_features,
        )

        self._built = True
        return self

    def create_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: Union[str, torch.dtype],
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Override to change to float32
        """
        return super(EnfugueTensorRTStableDiffusionPipeline, self).create_latents(
            batch_size, num_channels_latents, height, width, torch.float32, device, generator,
        )

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
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        if "clip" not in self.engine:
            return super(EnfugueTensorRTStableDiffusionPipeline, self).encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_2=prompt_2,
                negative_prompt_2=negative_prompt_2
            )
        
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

        # Tokenize prompt
        text_input_ids = (
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        text_input_ids_inp = self.device_view(text_input_ids)
        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt

        text_embeddings = (
            self.engine["clip"].infer({"input_ids": text_input_ids_inp}, self.stream)["text_embeddings"].clone()
        )

        # Tokenize negative prompt
        uncond_input_ids = (
            self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        uncond_input_ids_inp = self.device_view(uncond_input_ids)
        uncond_embeddings = self.engine["clip"].infer({"input_ids": uncond_input_ids_inp}, self.stream)[
            "text_embeddings"
        ]

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings

    def get_controlnet_conditioning_blocks(
        self,
        device: Union[str, torch.device],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor],
        conditioning_scale: float,
        added_cond_kwargs: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Executes the controlnet inference
        """
        if "controlnet" not in self.engine:
            return super(EnfugueTensorRTStableDiffusionPipeline, self).get_controlnet_conditioning_blocks(
                device=device,
                latents=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
            )
        if controlnet_cond is None:
            return None, None

        timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep
        conditioning_scale_tensor = torch.Tensor([conditioning_scale]).to(dtype=torch.float32, device=device)
        inference_kwargs = {
            "sample": self.device_view(latents),
            "timestep": self.device_view(timestep_float),
            "encoder_hidden_states": self.device_view(encoder_hidden_states),
            "controlnet_cond": self.device_view(controlnet_cond),
            "conditioning_scale": self.device_view(conditioning_scale_tensor),
        }

        inferred = self.engine["controlnet"].infer(inference_kwargs, self.stream)
        return [inferred[f"down_block_{i}"] for i in range(12)], inferred["mid_block"]

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
        Runs the RT engine inference on the unet
        """
        engine_name = "controlledunet"
        if engine_name not in self.engine:
            engine_name = "unet"
        if engine_name not in self.engine:
            return super(EnfugueTensorRTStableDiffusionPipeline, self).predict_noise_residual(
                latents=latents,
                timestep=timestep,
                embeddings=embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
            )

        timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

        inference_kwargs = {
            "sample": self.device_view(latents),
            "timestep": self.device_view(timestep_float),
            "encoder_hidden_states": self.device_view(embeddings),
        }

        if down_block_additional_residuals is not None and mid_block_additional_residual is not None:
            if engine_name == "unet":
                if not getattr(self, "_informed_of_bad_controlled_unet", False):
                    logger.error(
                        "Incorrect use of UNet/ControlledUNet models. Controlnet input was detected, but only the non-controlled UNet is loaded. Controlnet input will be ignored going forward."
                    )
                    self._informed_of_bad_controlled_unet = True
            else:
                inference_kwargs["mid_block"] = self.device_view(mid_block_additional_residual)
                for i, block in enumerate(down_block_additional_residuals):
                    inference_kwargs[f"down_block_{i}"] = self.device_view(block)
        return self.engine[engine_name].infer(inference_kwargs, self.stream)["latent"]
