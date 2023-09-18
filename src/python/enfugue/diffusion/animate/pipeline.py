# Inspired by https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py
from __future__ import annotations

import os
import torch

from typing import Optional, Dict, Any, Union, Callable, List, TYPE_CHECKING

from pibble.util.files import load_json

from diffusers.utils import WEIGHTS_NAME, DIFFUSERS_CACHE
from diffusers.models.modeling_utils import ModelMixin

from einops import rearrange

from enfugue.util import check_download_to_dir
from enfugue.diffusion.pipeline import EnfugueStableDiffusionPipeline
from enfugue.diffusion.animate.unet import UNet3DConditionModel # type: ignore[attr-defined]

if TYPE_CHECKING:
    from transformers import (
        CLIPTokenizer,
        CLIPTextModel,
        CLIPImageProcessor,
        CLIPTextModelWithProjection,
    )
    from diffusers.models import (
        AutoencoderKL,
        ControlNetModel,
        UNet2DConditionModel,
    )
    from diffusers.pipelines.stable_diffusion import (
        StableDiffusionSafetyChecker
    )
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from enfugue.diffusers.support.ip import IPAdapter
    from enfugue.diffusers.constants import MASK_TYPE_LITERAL

class EnfugueAnimateStableDiffusionPipeline(EnfugueStableDiffusionPipeline):
    unet_3d: Optional[UNet3DConditionModel]
    vae: AutoencoderKL

    STATIC_SCHEDULER_KWARGS = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear"
    }

    MOTION_MODULE_V2 = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
    MOTION_MODULE = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Optional[CLIPTextModel],
        text_encoder_2: Optional[CLIPTextModelWithProjection],
        tokenizer: Optional[CLIPTokenizer],
        tokenizer_2: Optional[CLIPTokenizer],
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetic_score: bool = False,
        force_full_precision_vae: bool = False,
        controlnets: Optional[Dict[str, ControlNetModel]] = None,
        ip_adapter: Optional[IPAdapter] = None,
        engine_size: int = 512,  # Recommended even for machines that can handle more
        chunking_size: int = 32,
        chunking_mask_type: MASK_TYPE_LITERAL = "bilinear",
        chunking_mask_kwargs: Dict[str, Any] = {},
        temporal_engine_size: int = 16,
        temporal_chunking_size: int = 4,
        override_scheduler_config: bool = True,
    ) -> None:
        super(EnfugueAnimateStableDiffusionPipeline, self).__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            force_full_precision_vae=force_full_precision_vae,
            requires_aesthetic_score=requires_aesthetic_score,
            controlnets=controlnets,
            ip_adapter=ip_adapter,
            engine_size=engine_size,
            chunking_size=chunking_size,
            chunking_mask_type=chunking_mask_type,
            chunking_mask_kwargs=chunking_mask_kwargs,
            temporal_engine_size=temporal_engine_size,
            temporal_chunking_size=temporal_chunking_size
        )
        if override_scheduler_config:
            self.scheduler_config = {
                **self.scheduler_config,
                **EnfugueAnimateStableDiffusionPipeline.STATIC_SCHEDULER_KWARGS
            }
            self.scheduler.register_to_config(
                **EnfugueAnimateStableDiffusionPipeline.STATIC_SCHEDULER_KWARGS
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs: Any
    ) -> EnfugueAnimateStableDiffusionPipeline:
        """
        Override from_pretrained to reload the unet as a 3D condition model instead.
        """
        pipe = super(EnfugueAnimateStableDiffusionPipeline, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        unet_dir = os.path.join(pretrained_model_name_or_path, "unet") # type: ignore[arg-type]
        unet_config = os.path.join(unet_dir, "config.json")
        unet_weights = os.path.join(unet_dir, WEIGHTS_NAME)
        
        if not os.path.exists(unet_config) or not os.path.exists(unet_weights):
            raise IOError(f"Couldn't find UNet config or weights.")

        unet = cls.create_unet(
            load_json(unet_config),
            kwargs.get("cache_dir", DIFFUSERS_CACHE),
        )
        unet.load_state_dict(torch.load(unet_weights), strict=False)
        if "torch_dtype" in kwargs:
            unet = unet.to(kwargs["torch_dtype"])
        pipe.unet = unet
        return pipe

    @classmethod
    def create_unet(
        cls,
        config: Dict[str, Any],
        cache_dir: str,
        use_mm_v2: bool = True,
        **unet_additional_kwargs: Any
    ) -> ModelMixin:
        """
        Creates the 3D Unet
        """
        config["_class_name"] = "UNet3DConditionModel"
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        unet_additional_kwargs["use_inflated_groupnorm"] = use_mm_v2
        unet_additional_kwargs["unet_use_cross_frame_attention"] = False
        unet_additional_kwargs["unet_use_temporal_attention"] = False
        unet_additional_kwargs["use_motion_module"] = True
        unet_additional_kwargs["motion_module_resolutions"] = [1, 2, 4, 8]
        unet_additional_kwargs["motion_module_mid_block"] = use_mm_v2
        unet_additional_kwargs["motion_module_decoder_only"] = False
        unet_additional_kwargs["motion_module_type"] = "Vanilla"
        unet_additional_kwargs["motion_module_kwargs"] = {
            "num_attention_heads": 8,
            "num_transformer_block": 1,
            "attention_block_types": [
                "Temporal_Self",
                "Temporal_Self"
            ],
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 32 if use_mm_v2 else 24,
            "temporal_attention_dim_div": 1
        }

        model = UNet3DConditionModel.from_config(config, **unet_additional_kwargs)
        motion_module = cls.MOTION_MODULE_V2 if use_mm_v2 else cls.MOTION_MODULE
        model_file = check_download_to_dir(motion_module, cache_dir)
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model

    def decode_latents(
        self,
        latents: torch.Tensor,
        device: Union[str, torch.device],
        progress_callback: Optional[Callable[[bool], None]] = None
    ) -> torch.Tensor:
        """
        Decodes each video frame individually.
        """
        animation_frames = latents.shape[2]
        #latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        dtype = latents.dtype
        # Force full precision VAE
        #self.vae = self.vae.to(torch.float32)
        #latents = latents.to(torch.float32)
        video: List[torch.Tensor] = []
        for frame_index in range(latents.shape[0]):
            video.append(
                super(EnfugueAnimateStableDiffusionPipeline, self).decode_latents(
                    latents=latents[frame_index:frame_index+1],
                    device=device,
                    progress_callback=progress_callback
                )
            )
        video = torch.cat(video) # type: ignore
        video = rearrange(video, "(b f) c h w -> b c f h w", f = animation_frames) # type: ignore
        video = (video / 2 + 0.5).clamp(0, 1) # type: ignore
        video = video.cpu().float() # type: ignore
        #self.vae.to(dtype)
        return video # type: ignore
