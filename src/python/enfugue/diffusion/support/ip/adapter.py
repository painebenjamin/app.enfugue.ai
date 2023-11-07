from __future__ import annotations

from typing import List, Union, Dict, Any, Iterator, Optional, Tuple, Callable, TYPE_CHECKING
from typing_extensions import Self
from contextlib import contextmanager
from enfugue.util import logger
from enfugue.diffusion.support.model import SupportModel

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from enfugue.diffusion.constants import IP_ADAPTER_LITERAL
    from enfugue.diffusion.support.ip.projection import ImageProjectionModel
    from enfugue.diffusion.support.ip.resampler import Resampler # type: ignore
    from diffusers.models import UNet2DConditionModel, ControlNetModel
    from transformers import (
        CLIPVisionModelWithProjection,
        CLIPImageProcessor,
        PretrainedConfig
    )

class IPAdapter(SupportModel):
    """
    Modifies the tencent IP adapter so it can load/unload at will
    """
    cross_attention_dim: int = 768
    is_sdxl: bool = False
    model: IP_ADAPTER_LITERAL = "default"

    DEFAULT_ENCODER_CONFIG_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json"
    DEFAULT_ENCODER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin"
    DEFAULT_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin"

    FINE_GRAINED_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin"
    FACE_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin"
    
    XL_ENCODER_CONFIG_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/config.json"
    XL_ENCODER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/pytorch_model.bin"
    XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin"

    FINE_GRAINED_XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
    FACE_XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"

    def load(
        self,
        unet: UNet2DConditionModel,
        model: Optional[IP_ADAPTER_LITERAL]="default",
        is_sdxl: bool=False,
        scale: float=1.0,
        keepalive_callback: Optional[Callable[[],None]]=None,
        controlnets: Optional[Dict[str, ControlNetModel]]=None,
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

        if model is None:
            model = "default"

        if (
            self.cross_attention_dim != unet.config.cross_attention_dim or # type: ignore[attr-defined]
            self.is_sdxl != is_sdxl or 
            self.model != model
        ):
            del self.projector
            del self.encoder

        self.is_sdxl = is_sdxl
        self.model = model
        self.cross_attention_dim = unet.config.cross_attention_dim # type: ignore[attr-defined]

        self._default_unet_attention_processors: Dict[str, Any] = {}
        self._default_controlnet_attention_processors: Dict[str, Dict[str, Any]] = {}

        new_attention_processors: Dict[str, Any] = {}

        for name in unet.attn_processors.keys():
            current_processor = unet.attn_processors[name]
            cross_attention_dim = None if name.endswith("attn1.processor") else self.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1] # type: ignore[attr-defined]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id] # type: ignore[attr-defined]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id] # type: ignore[attr-defined]

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
                    scale=scale,
                    num_tokens=self.tokens,
                ).to(self.device, dtype=self.dtype)

        keepalive_callback()
        unet.set_attn_processor(new_attention_processors)
        layers = torch.nn.ModuleList(unet.attn_processors.values()) # type: ignore[arg-type]

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

    @property
    def use_fine_grained(self) -> bool:
        """
        Returns true if using a plus model
        """
        return self.model == "plus" or self.model == "plus-face"

    def check_download(
        self,
        is_sdxl: bool=False,
        model: Optional[IP_ADAPTER_LITERAL]="default",
        task_callback: Optional[Callable[[str], None]]=None,
    ) -> None:
        """
        Downloads necessary files for any pipeline
        """
        # Gather previous state
        _task_callback = self.task_callback
        _is_sdxl = self.is_sdxl
        _model = self.model

        # Set new state
        self.task_callback = task_callback
        self.is_sdxl = is_sdxl
        if model is None:
            self.model = "default"
        else:
            self.model = model

        # Trigger getters
        if is_sdxl:
            _ = self.xl_encoder_config
            _ = self.xl_encoder_model
            _ = self.xl_image_prompt_checkpoint
        else:
            _ = self.default_encoder_config
            _ = self.default_encoder_model
            _ = self.default_image_prompt_checkpoint

        # Reset state
        self.task_callback = _task_callback
        self.is_sdxl = _is_sdxl
        self.model = _model

    def set_scale(
        self,
        unet: UNet2DConditionModel,
        scale: float,
        is_sdxl: bool=False,
        model: Optional[IP_ADAPTER_LITERAL]="default",
        keepalive_callback: Optional[Callable[[],None]]=None,
        controlnets: Optional[Dict[str, ControlNetModel]]=None,
    ) -> int:
        """
        Sets the scale on attention processors.
        """
        if model is None:
            model = "default"
        if self.is_sdxl != is_sdxl or self.model != model:
            # Completely reload adapter
            self.unload(unet, controlnets)
            self.load(
                unet,
                is_sdxl=is_sdxl,
                scale=scale,
                model=model,
                keepalive_callback=keepalive_callback,
                controlnets=controlnets
            )
            return 1
        from enfugue.diffusion.support.ip.attention import ( # type: ignore[attr-defined]
            IPAttentionProcessor,
            IPAttentionProcessor2_0,
        )
        processors_altered = 0
        for name in unet.attn_processors.keys():
            processor = unet.attn_processors[name]
            if isinstance(processor, IPAttentionProcessor) or isinstance(processor, IPAttentionProcessor2_0):
                processor.scale = scale
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
    def default_encoder_model(self) -> str:
        """
        Gets the path to the IP model for 1.5
        Downloads if needed
        """
        return self.get_model_file(
            self.DEFAULT_ENCODER_PATH,
            filename="ip-adapter_sd15_encoder.pth",
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def default_encoder_config(self) -> str:
        """
        Gets the path to the IP model for 1.5
        Downloads if needed
        """
        return self.get_model_file(
            self.DEFAULT_ENCODER_CONFIG_PATH,
            filename="ip-adapter_sd15_encoder_config.json"
        )

    @property
    def default_image_prompt_checkpoint(self) -> str:
        """
        Gets the path to the IP checkpoint for 1.5
        Downloads if needed
        """
        if self.model == "plus-face":
            model_url = self.FACE_ADAPTER_PATH
            filename = "ip-adapter-plus-face_sd15.pth"
        elif self.model == "plus":
            model_url = self.FINE_GRAINED_ADAPTER_PATH
            filename = "ip-adapter-plus_sd15.pth"
        else:
            model_url = self.DEFAULT_ADAPTER_PATH
            filename = "ip-adapter_sd15.pth"

        return self.get_model_file(
            model_url,
            filename=filename,
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def xl_encoder_model(self) -> str:
        """
        Gets the path to the IP model for 1.5
        Downloads if needed
        """
        if self.use_fine_grained:
            return self.default_encoder_model

        return self.get_model_file(
            self.XL_ENCODER_PATH,
            filename="ip-adapter_sdxl_encoder.pth",
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def xl_encoder_config(self) -> str:
        """
        Gets the path to the IP model for 1.5
        Downloads if needed
        """
        if self.use_fine_grained:
            return self.default_encoder_config

        return self.get_model_file(
            self.XL_ENCODER_CONFIG_PATH,
            filename="ip-adapter_sdxl_encoder_config.json"
        )

    @property
    def xl_image_prompt_checkpoint(self) -> str:
        """
        Gets the path to the IP checkpoint for XL
        Downloads if needed
        """
        if self.model == "plus-face":
            model_url = self.FACE_XL_ADAPTER_PATH
            filename = "ip-adapter-plus-face_sdxl_vit-h.pth"
        elif self.model == "plus":
            model_url = self.FINE_GRAINED_XL_ADAPTER_PATH
            filename = "ip-adapter-plus_sdxl_vit-h.pth"
        else:
            model_url = self.XL_ADAPTER_PATH
            filename = "ip-adapter_sdxl.pth"
            
        return self.get_model_file(
            model_url,
            filename=filename,
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def tokens(self) -> int:
        """
        Gets the number of tokens for extra clip context
        """
        return getattr(self, "_tokens", 16 if self.use_fine_grained else 4)

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
        from transformers import (
            CLIPVisionModelWithProjection,
            PretrainedConfig
        )
        if not hasattr(self, "_encoder"):
            if self.is_sdxl:
                logger.debug(f"Initializing CLIPVisionModelWithProjection from {self.xl_encoder_model}")
                self._encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.xl_encoder_model,
                    config=PretrainedConfig.from_json_file(self.xl_encoder_config),
                    use_safetensors="safetensors" in self.xl_encoder_model,
                )
            else:
                logger.debug(f"Initializing CLIPVisionModelWithProjection from {self.default_encoder_model}")
                self._encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.default_encoder_model,
                    config=PretrainedConfig.from_json_file(self.default_encoder_config),
                    use_safetensors="safetensors" in self.default_encoder_model,
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
        from enfugue.diffusion.util.torch_util import load_state_dict
        logger.debug(f"Loading state dictionary from {self.default_image_prompt_checkpoint}")
        return load_state_dict(self.default_image_prompt_checkpoint)

    @property
    def xl_state_dict(self) -> Dict[str, Any]:
        """
        Gets the state dict from the IP checkpoint
        """
        from enfugue.diffusion.util.torch_util import load_state_dict
        logger.debug(f"Loading state dictionary from {self.xl_image_prompt_checkpoint}")
        return load_state_dict(self.xl_image_prompt_checkpoint)

    @property
    def projector(self) -> Union[ImageProjectionModel, Resampler]:
        """
        Gets the projection model
        """
        if not hasattr(self, "_projector"):
            logger.debug(f"Initializing ImageProjectionModel with cross-attention dimensions of {self.cross_attention_dim}")
            if self.use_fine_grained:
                from enfugue.diffusion.support.ip.resampler import Resampler # type: ignore[attr-defined]
                self._projector = Resampler(
                    dim=1280 if self.is_sdxl else self.cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=20 if self.is_sdxl else 12,
                    num_queries=self.tokens,
                    embedding_dim=self.encoder.config.hidden_size,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4
                )
            else:
                from enfugue.diffusion.support.ip.projection import ImageProjectionModel
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
        from transformers import CLIPImageProcessor
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

            clip_image = clip_image.to(self.device, dtype=self.dtype)
            clip_image_encoded = self.encoder(
                clip_image,
                output_hidden_states=self.use_fine_grained
            )

            if self.use_fine_grained:
                clip_image_embeds = clip_image_encoded.hidden_states[-2]
            else:
                clip_image_embeds = clip_image_encoded.image_embeds

            image_prompt_embeds = self.projector(clip_image_embeds)

            if self.use_fine_grained:
                image_uncond_prompt_input = self.encoder(
                    torch.zeros_like(clip_image),
                    output_hidden_states=True
                ).hidden_states[-2]
            else:
                image_uncond_prompt_input = torch.zeros_like(clip_image_embeds)

            image_uncond_prompt_embeds = self.projector(image_uncond_prompt_input)
            return image_prompt_embeds, image_uncond_prompt_embeds
