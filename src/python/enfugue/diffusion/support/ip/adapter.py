from __future__ import annotations

import os

from typing import List, Union, Dict, Any, Iterator, Optional, Tuple, Callable, TYPE_CHECKING
from typing_extensions import Self
from contextlib import contextmanager

from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from enfugue.util import check_download_to_dir, logger
from enfugue.diffusion.support.model import SupportModel

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from enfugue.diffusion.support.ip.projection import ImageProjectionModel
    from diffusers.models import UNet2DConditionModel, ControlNetModel

class IPAdapter(SupportModel):
    """
    Modifies the tencent IP adapter so it can load/unload at will
    """
    cross_attention_dim: int = 768
    is_sdxl: bool = False

    DEFAULT_ENCODER_PATH = (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
    )

    DEFAULT_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin"
    
    XL_ENCODER_PATH = (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/config.json",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/pytorch_model.bin",
    )

    XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin"

    def load(
        self,
        unet: UNet2DConditionModel,
        is_sdxl: bool = False,
        scale: float = 1.0,
        keepalive_callback: Optional[Callable[[],None]] = None,
        controlnets: Optional[Dict[str, ControlNetModel]] = None,
    ) -> None:
        """
        Loads the IP adapter.
        """
        if keepalive_callback is None:
            keepalive_callback = lambda: None
        import torch
        from diffusers.models.attention_processor import (
            AttnProcessor2_0,
            LoRAAttnProcessor2_0
        )
        from enfugue.diffusion.support.ip.attention import ( # type: ignore[attr-defined]
            CNAttentionProcessor,
            CNAttentionProcessor2_0,
            IPAttentionProcessor,
            IPAttentionProcessor2_0,
            AttentionProcessor,
            AttentionProcessor2_0,
        )
        if self.cross_attention_dim != unet.config.cross_attention_dim or self.is_sdxl != is_sdxl:
            del self.projector
            del self.encoder

        self.is_sdxl = is_sdxl
        self.cross_attention_dim = unet.config.cross_attention_dim

        self._default_unet_attention_processors: Dict[str, Any] = {}
        self._default_controlnet_attention_processors: Dict[str, Dict[str, Any]] = {}

        new_attention_processors: Dict[str, Any] = {}

        for name in unet.attn_processors.keys():
            current_processor = unet.attn_processors[name]
            cross_attention_dim = None if name.endswith("attn1.processor") else self.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            self._default_unet_attention_processors[name] = current_processor

            if cross_attention_dim is None:
                if type(current_processor) in [AttnProcessor2_0, LoRAAttnProcessor2_0]:
                    attn_class = AttentionProcessor2_0
                else:
                    attn_class = AttentionProcessor
                new_attention_processors[name] = attn_class()
            else:
                if type(current_processor) in [AttnProcessor2_0, LoRAAttnProcessor2_0]:
                    ip_attn_class = IPAttentionProcessor2_0
                else:
                    ip_attn_class = IPAttentionProcessor
                new_attention_processors[name] = ip_attn_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale
                ).to(self.device, dtype=self.dtype)

        keepalive_callback()
        unet.set_attn_processor(new_attention_processors)
        layers = torch.nn.ModuleList(unet.attn_processors.values())
        state_dict = self.xl_state_dict if is_sdxl else self.default_state_dict

        keepalive_callback()
        layers.load_state_dict(state_dict["ip_adapter"])

        keepalive_callback()
        self.projector.load_state_dict(state_dict["image_proj"])

        if controlnets is not None:
            keepalive_callback()
            for controlnet in controlnets:
                new_processors = {}
                current_processors = controlnets[controlnet].attn_processors
                self._default_controlnet_attention_processors[controlnet] = {}
                for key in current_processors:
                    self._default_controlnet_attention_processors[controlnet][key] = current_processors[key]
                    if isinstance(current_processors[key], AttnProcessor2_0):
                        new_processors[key] = CNAttentionProcessor2_0()
                    else:
                        new_processors[key] = CNAttentionProcessor()
                controlnets[controlnet].set_attn_processor(new_processors)

    def set_scale(self, unet: UNet2DConditionModel, new_scale: float) -> int:
        """
        Sets the scale on attention processors.
        """
        from enfugue.diffusion.support.ip.attention import ( # type: ignore[attr-defined]
            IPAttentionProcessor,
            IPAttentionProcessor2_0,
        )
        processors_altered = 0
        for name in unet.attn_processors.keys():
            processor = unet.attn_processors[name]
            if isinstance(processor, IPAttentionProcessor) or isinstance(processor, IPAttentionProcessor2_0):
                processor.scale = new_scale
                processors_altered += 1
        return processors_altered

    def unload(self, unet: UNet2DConditionModel, controlnets: Optional[Dict[str, ControlNetModel]] = None) -> None:
        """
        Unloads the IP adapter by resetting attention processors to previous values
        """
        if not hasattr(self, "_default_unet_attention_processors"):
            raise RuntimeError("IP adapter was not loaded, cannot unload")

        unet.set_attn_processor({**self._default_unet_attention_processors})

        if controlnets is not None:
            for controlnet in controlnets:
                if controlnet in self._default_controlnet_attention_processors:
                    controlnets[controlnet].set_attn_processor(
                        {**self._default_controlnet_attention_processors[controlnet]}
                    )

        del self._default_unet_attention_processors
        del self._default_controlnet_attention_processors

    @property
    def adapter_directory(self) -> str:
        """
        Gets the location where adapter models will be downloaded
        """
        path = os.path.join(self.model_dir, "ip-adapter")
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    @property
    def default_adapter_directory(self) -> str:
        """
        Gets the path to where the 1.5 adapter models are stored.
        Downloads if needed
        """
        if not hasattr(self, "_default_adapter_directory"):
            config_path, bin_path = self.DEFAULT_ENCODER_PATH
            path = os.path.join(self.adapter_directory, "default")
            if not os.path.isdir(path):
                os.makedirs(path)
            check_download_to_dir(config_path, path)
            check_download_to_dir(bin_path, path)
            self._default_adapter_directory = path
        return self._default_adapter_directory

    @property
    def default_image_prompt_checkpoint(self) -> str:
        """
        Gets the path to the IP checkpoint for 1.5
        Downloads if needed
        """
        if not hasattr(self, "_default_ip_checkpoint"):
            self._default_ip_checkpoint = check_download_to_dir(self.DEFAULT_ADAPTER_PATH, self.adapter_directory)
        return self._default_ip_checkpoint
    
    @property
    def xl_adapter_directory(self) -> str:
        """
        Gets the path to where the XL adapter models are stored.
        Downloads if needed
        """
        if not hasattr(self, "_xl_adapter_directory"):
            config_path, bin_path = self.XL_ENCODER_PATH
            path = os.path.join(self.adapter_directory, "xl")
            if not os.path.isdir(path):
                os.makedirs(path)
            check_download_to_dir(config_path, path)
            check_download_to_dir(bin_path, path)
            self._xl_adapter_directory = path
        return self._xl_adapter_directory

    @property
    def xl_image_prompt_checkpoint(self) -> str:
        """
        Gets the path to the IP checkpoint for XL
        Downloads if needed
        """
        if not hasattr(self, "_xl_ip_checkpoint"):
            self._xl_ip_checkpoint = check_download_to_dir(self.XL_ADAPTER_PATH, self.adapter_directory)
        return self._xl_ip_checkpoint

    @property
    def tokens(self) -> int:
        """
        Gets the number of tokens for extra clip context
        """
        return getattr(self, "_tokens", 4)

    @tokens.setter
    def tokens(self, amount: int) -> None:
        """
        Sets the token amount
        """
        self._tokens = amount

    @property
    def encoder(self) -> CLIPVisionModelWithProjection:
        """
        Gets the encoder, initializes if needed
        """
        if not hasattr(self, "_encoder"):
            if self.is_sdxl:
                logger.debug(f"Initializing CLIPVisionModelWithProjection from {self.xl_adapter_directory}")
                self._encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.xl_adapter_directory
                )
            else:
                logger.debug(f"Initializing CLIPVisionModelWithProjection from {self.default_adapter_directory}")
                self._encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.default_adapter_directory
                )
        return self._encoder

    @encoder.deleter
    def encoder(self) -> None:
        """
        Deletes the encoder if it exists.
        """
        if hasattr(self, "_encoder"):
            del self._encoder

    @property
    def default_state_dict(self) -> Dict[str, Any]:
        """
        Gets the state dict from the IP checkpoint
        """
        if not hasattr(self, "_default_tate_dict"):
            import torch
            self._default_state_dict = torch.load(self.default_image_prompt_checkpoint, map_location="cpu")
        return self._default_state_dict

    @default_state_dict.deleter
    def default_state_dict(self) -> None:
        """
        Deletes the state dict to clear memory
        """
        if hasattr(self, "_default_state_dict"):
            del self._default_state_dict
    
    @property
    def xl_state_dict(self) -> Dict[str, Any]:
        """
        Gets the state dict from the IP checkpoint
        """
        if not hasattr(self, "_xl_tate_dict"):
            import torch
            self._xl_state_dict = torch.load(self.xl_image_prompt_checkpoint, map_location="cpu")
        return self._xl_state_dict

    @xl_state_dict.deleter
    def xl_state_dict(self) -> None:
        """
        Deletes the state dict to clear memory
        """
        if hasattr(self, "_xl_state_dict"):
            del self._xl_state_dict

    @property
    def projector(self) -> ImageProjectionModel:
        """
        Gets the projection model
        """
        if not hasattr(self, "_projector"):
            from enfugue.diffusion.support.ip.projection import ImageProjectionModel
            logger.debug(f"Initializing ImageProjectionModel with cross-attention dimensions of {self.cross_attention_dim}")
            self._projector = ImageProjectionModel(
                clip_embeddings_dim=self.encoder.config.projection_dim,
                cross_attention_dim=self.cross_attention_dim,
                clip_extra_context_tokens=self.tokens
            )
        return self._projector

    @projector.deleter
    def projector(self) -> None:
        """
        Delete the projection model
        """
        if hasattr(self, "_projector"):
            del self._projector

    @property
    def processor(self) -> CLIPImageProcessor:
        """
        Gets the processor, initializes if needed
        """
        if not hasattr(self, "_processor"):
            self._processor = CLIPImageProcessor()
        return self._processor

    @contextmanager
    def context(self) -> Iterator[Self]:
        """
        Override parent context to send models to device
        """
        import torch
        with super(IPAdapter, self).context():
            self.encoder.to(device=self.device, dtype=self.dtype)
            self.projector.to(device=self.device, dtype=self.dtype)
            with torch.inference_mode():
                yield self
        self.to("cpu")

    def to(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None
    ) -> IPAdapter:
        """
        Sends the encoder and project to a diff device
        """
        kwargs: Dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if hasattr(self, "_encoder"):
            self.encoder.to(**kwargs)
        if hasattr(self, "_projector"):
            self.projector.to(**kwargs) # type: ignore[call-overload]
        return self

    def probe(
        self,
        images: Union[Image, List[Image]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Probes an image by interrogating prompt embeds
        """
        import torch
        with self.context():
            if not isinstance(images, list):
                images = [images]

            clip_image = self.processor(
                images=images,
                return_tensors="pt"
            ).pixel_values

            clip_image_embeds = self.encoder(
                clip_image.to(
                    device=self.device,
                    dtype=self.dtype
                )
            ).image_embeds

            image_prompt_embeds = self.projector(clip_image_embeds)
            image_uncond_prompt_embeds = self.projector(torch.zeros_like(clip_image_embeds))

            return image_prompt_embeds, image_uncond_prompt_embeds
