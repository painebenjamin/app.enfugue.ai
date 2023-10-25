# Inspired by https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from typing import Optional, Dict, Any, Union, Callable, List, TYPE_CHECKING

from pibble.util.files import load_json

from diffusers.utils import WEIGHTS_NAME, DIFFUSERS_CACHE
from diffusers.models.modeling_utils import ModelMixin

from einops import rearrange

from enfugue.util import check_download_to_dir, find_file_in_directory
from enfugue.diffusion.pipeline import EnfugueStableDiffusionPipeline

from enfugue.diffusion.animate.diff.unet import UNet3DConditionModel as AnimateDiffUNet # type: ignore[attr-defined]
from enfugue.diffusion.animate.hotshot.unet import UNet3DConditionModel as HotshotUNet # type: ignore[attr-defined]

if TYPE_CHECKING:
    from transformers import (
        CLIPTokenizer,
        CLIPTextModel,
        CLIPImageProcessor,
        CLIPTextModelWithProjection,
    )
    from diffusers.models import (
        AutoencoderKL,
        AutoencoderTiny,
        ControlNetModel,
        UNet2DConditionModel,
    )
    from diffusers.pipelines.stable_diffusion import (
        StableDiffusionSafetyChecker
    )
    from diffusers.schedulers import KarrasDiffusionSchedulers
    from enfugue.diffusion.support.ip import IPAdapter
    from enfugue.diffusion.util import Chunker, MaskWeightBuilder
    from enfugue.diffusion.constants import MASK_TYPE_LITERAL

class EnfugueAnimateStableDiffusionPipeline(EnfugueStableDiffusionPipeline):
    unet_3d: Optional[Union[AnimateDiffUNet, HotshotUNet]]
    vae: AutoencoderKL

    STATIC_SCHEDULER_KWARGS = {
        "timestep_spacing": "leading",
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.011,
        "beta_schedule": "linear"
    }

    HOTSHOT_XL_PATH = "hotshotco/Hotshot-XL"
    MOTION_MODULE_V2 = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
    MOTION_MODULE = "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt"

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
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetic_score: bool = False,
        force_full_precision_vae: bool = False,
        controlnets: Optional[Dict[str, ControlNetModel]] = None,
        ip_adapter: Optional[IPAdapter] = None,
        engine_size: Optional[int] = 512,  # Recommended even for machines that can handle more
        chunking_size: Optional[int] = 32,
        chunking_mask_type: MASK_TYPE_LITERAL = "bilinear",
        chunking_mask_kwargs: Dict[str, Any] = {},
        temporal_engine_size: Optional[int] = 16,
        temporal_chunking_size: Optional[int] = 4,
        override_scheduler_config: bool = True,
    ) -> None:
        super(EnfugueAnimateStableDiffusionPipeline, self).__init__(
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
        motion_module: Optional[str] = None,
        **kwargs: Any
    ) -> EnfugueAnimateStableDiffusionPipeline:
        """
        Override from_pretrained to reload the unet as a 3D condition model instead.
        """
        pipe = super(EnfugueAnimateStableDiffusionPipeline, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)
        unet_dir = os.path.join(pretrained_model_name_or_path, "unet") # type: ignore[arg-type]
        unet_config = os.path.join(unet_dir, "config.json")
        unet_weights = os.path.join(unet_dir, WEIGHTS_NAME)

        if not os.path.exists(unet_config):
            raise IOError(f"Couldn't find UNet config at {unet_config}")
        if not os.path.exists(unet_weights):
            # Check for safetensors version
            safetensors_weights = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
            if os.path.exists(safetensors_weights):
                unet_weights = safetensors_weights
            else:
                raise IOError(f"Couldn't find UNet weights at {unet_weights} or {safetensors_weights}")

        from enfugue.diffusion.util.torch_util import load_state_dict
        unet = cls.create_unet(
            load_json(unet_config),
            kwargs.get("cache_dir", DIFFUSERS_CACHE),
            motion_module=motion_module
        )

        state_dict = load_state_dict(unet_weights)
        unet.load_state_dict(state_dict, strict=False)

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
        motion_module: Optional[str] = None,
        task_callback: Optional[Callable[[str], None]]=None,
        **unet_additional_kwargs: Any
    ) -> ModelMixin:
        """
        Creates the 3D Unet
        """
        if config.get("sample_size", 64) == 128:
            # SDXL, instantiate Hotshot XL UNet
            return cls.create_hotshot_unet(
                config=config,
                cache_dir=cache_dir,
                task_callback=task_callback,
                **unet_additional_kwargs
            )
        return cls.create_diff_unet(
            config=config,
            cache_dir=cache_dir,
            use_mm_v2=use_mm_v2,
            motion_module=motion_module,
            task_callback=task_callback,
            **unet_additional_kwargs
        )

    @classmethod
    def create_hotshot_unet(
        cls,
        config: Dict[str, Any],
        cache_dir: str,
        task_callback: Optional[Callable[[str], None]]=None,
        position_encoder_truncate_length: Optional[int]=None,
        position_encoder_scale_length: Optional[int]=None,
        **unet_additional_kwargs: Any
    ) -> ModelMixin:
        """
        Creates a UNet3DConditionModel then loads hotshot into it
        """
        config["_class_name"] = "UNet3DConditionModel"
        config["down_block_types"] = [
            "DownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
        ]
        config["up_block_types"] = [
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "UpBlock3D"
        ]
        if position_encoder_scale_length:
            config["positional_encoding_max_length"] = position_encoder_scale_length

        # Instantiate from 2D model config
        model = HotshotUNet.from_config(config)

        # Get base model
        if task_callback is not None:
            if not os.path.exists(
                os.path.join(
                    cache_dir,
                    cls.HOTSHOT_XL_PATH.replace("/", "--"),
                    "config.json"
                )
            ):
                task_callback(f"Downloading Repository {cls.HOTSHOT_XL_PATH}")

        # Instantiate pretrained model
        hotshot_unet = HotshotUNet.from_pretrained(
            cls.HOTSHOT_XL_PATH,
            subfolder="unet",
            cache_dir=cache_dir,
        )

        # Load base model into transformed 2d
        hotshot_state_dict = hotshot_unet.state_dict()
        for key in list(hotshot_state_dict.keys()):
            if "temporal" not in key:
                hotshot_state_dict.pop(key)
            elif key.endswith(".positional_encoding"):
                if position_encoder_truncate_length is not None:
                    hotshot_state_dict[key] = hotshot_state_dict[key][:, :position_encoder_truncate_length]
                if position_encoder_scale_length is not None:
                    tensor_shape = hotshot_state_dict[key].shape
                    tensor = rearrange(hotshot_state_dict[key], "(t b) f d -> t b f d", t=1)
                    tensor = F.interpolate(tensor, size=(position_encoder_scale_length, tensor_shape[-1]), mode="bilinear")
                    hotshot_state_dict[key] = rearrange(tensor, "t b f d -> (t b) f d")
                    del tensor

        model.load_state_dict(hotshot_state_dict, strict=False)
        del hotshot_state_dict
        del hotshot_unet
        return model

    @classmethod
    def create_diff_unet(
        cls,
        config: Dict[str, Any],
        cache_dir: str,
        use_mm_v2: bool=True,
        motion_module: Optional[str]=None,
        task_callback: Optional[Callable[[str], None]]=None,
        position_encoder_truncate_length: Optional[int]=None,
        position_encoder_scale_length: Optional[int]=None,
        **unet_additional_kwargs: Any
    ) -> ModelMixin:
        """
        Creates a UNet3DConditionModel then loads MM into it
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
        default_position_encoder_len = 32 if use_mm_v2 else 24
        position_encoder_len = default_position_encoder_len
        if position_encoder_scale_length:
            position_encoder_len = position_encoder_scale_length

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
            "temporal_position_encoding_max_len": position_encoder_len,
            "temporal_attention_dim_div": 1
        }

        model = AnimateDiffUNet.from_config(config, **unet_additional_kwargs)
        if not motion_module:
            motion_module = cls.MOTION_MODULE_V2 if use_mm_v2 else cls.MOTION_MODULE
            if task_callback is not None:
                if not os.path.exists(os.path.join(cache_dir, motion_module)):
                    task_callback(f"Downloading {motion_module}")
            motion_module = check_download_to_dir(motion_module, cache_dir)

        from enfugue.diffusion.util.torch_util import load_state_dict
        state_dict = load_state_dict(motion_module)

        if position_encoder_truncate_length is not None or position_encoder_scale_length is not None:
            for key in state_dict:
                if key.endswith(".pe"):
                    if position_encoder_truncate_length is not None:
                        state_dict[key] = state_dict[key][:, :position_encoder_truncate_length]
                    if position_encoder_scale_length is not None:
                        tensor_shape = state_dict[key].shape
                        tensor = rearrange(state_dict[key], "(t b) f d -> t b f d", t=1)
                        tensor = F.interpolate(tensor, size=(position_encoder_scale_length, tensor_shape[-1]), mode="bilinear")
                        state_dict[key] = rearrange(tensor, "t b f d -> (t b) f d")
                        del tensor

        model.load_state_dict(state_dict, strict=False)
        del state_dict

        return model

    def load_lora_weights(
        self,
        model: str,
        multiplier: float = 1.0,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any
    ) -> None:
        """
        Always use flexible loader
        """
        return self.load_flexible_lora_weights(
            model,
            multiplier,
            dtype
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        device: Union[str, torch.device],
        weight_builder: MaskWeightBuilder,
        chunker: Chunker,
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
                    weight_builder=weight_builder,
                    chunker=chunker,
                    progress_callback=progress_callback
                )
            )
        video = torch.cat(video) # type: ignore
        video = rearrange(video, "(b f) c h w -> b c f h w", f = animation_frames) # type: ignore
        video = (video / 2 + 0.5).clamp(0, 1) # type: ignore
        video = video.cpu().float() # type: ignore
        #self.vae.to(dtype)
        return video # type: ignore
