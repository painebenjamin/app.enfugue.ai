from __future__ import annotations

from typing import List, Union, Dict, Any, Iterator, Optional, Tuple, Callable, TYPE_CHECKING
from typing_extensions import Self
from contextlib import contextmanager
from enfugue.util import logger
from enfugue.diffusion.support.model import SupportModel
from pibble.util.files import load_json

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from enfugue.diffusion.constants import IP_ADAPTER_LITERAL
    from enfugue.diffusion.support.face import (
        FaceAnalyzer,
        FaceAnalyzerImageProcessor
    )
    from enfugue.diffusion.support.ip.projection import ImageProjectionModel
    from enfugue.diffusion.support.ip.resampler import Resampler # type: ignore
    from enfugue.diffusion.support.ip.attention import ( # type: ignore[attr-defined]
        AttentionProcessor,
        AttentionProcessor2_0,
        IPAttentionProcessor,
        IPAttentionProcessor2_0,
    )
    from diffusers.models import UNet2DConditionModel, ControlNetModel
    from transformers import (
        CLIPVisionModelWithProjection,
        CLIPImageProcessor,
    )

class IPAdapter(SupportModel):
    """
    Modifies the tencent IP adapter so it can load/unload at will
    """
    scale: float = 1.0
    cross_attention_dim: int = 768
    lora_rank: int = 128
    is_sdxl: bool = False
    model: IP_ADAPTER_LITERAL = "default"

    DEFAULT_ENCODER_CONFIG_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json"
    DEFAULT_ENCODER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin"
    XL_ENCODER_CONFIG_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/config.json"
    XL_ENCODER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/pytorch_model.bin"

    DEFAULT_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin"
    FINE_GRAINED_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin"
    FACE_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin"
    FACE_FULL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.bin"
    FACE_ID_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"
    FACE_ID_PLUS_PATH = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin"
    FACE_ID_PORTRAIT_PATH = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin"

    FACE_ID_ENCODER_PATH = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/model.safetensors"
    FACE_ID_ENCODER_CONFIG_PATH = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/config.json"

    XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin"
    FINE_GRAINED_XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
    FACE_XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"
    FACE_ID_XL_ADAPTER_PATH = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin"
    FACE_ID_XL_PLUS_PATH = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin"

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
            LoRAAttentionProcessor,
            LoRAIPAttentionProcessor,
            FaceIDLoRAAttentionProcessor,
            FaceIDLoRAAttentionProcessor2_0,
            FaceIDLoRAIPAttentionProcessor,
            FaceIDLoRAIPAttentionProcessor2_0
        )
        from enfugue.diffusion.animate.diff.sparse_controlnet import SparseControlNetModel # type: ignore[attr-defined]

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
        self.scale = scale
        self.cross_attention_dim = unet.config.cross_attention_dim # type: ignore[attr-defined]

        new_attention_processors: Dict[str, Any] = {}

        for name in unet.attn_processors.keys():
            current_processor = unet.attn_processors[name]
            use_torch_2 = type(current_processor) in [AttnProcessor2_0, LoRAAttnProcessor2_0]
            cross_attention_dim = None if name.endswith("attn1.processor") else self.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1] # type: ignore[attr-defined]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id] # type: ignore[attr-defined]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id] # type: ignore[attr-defined]

            processor_kwargs: Dict[str, Any] = {}
            if cross_attention_dim is None:
                if self.use_face_id:
                    if use_torch_2:
                        attn_class = FaceIDLoRAAttentionProcessor2_0
                    else:
                        attn_class = FaceIDLoRAAttentionProcessor
                    processor_kwargs["hidden_size"] = hidden_size
                    processor_kwargs["cross_attention_dim"] = cross_attention_dim
                    processor_kwargs["rank"] = self.lora_rank
                elif use_torch_2:
                    attn_class = AttentionProcessor2_0
                else:
                    attn_class = AttentionProcessor
                new_attention_processors[name] = attn_class(**processor_kwargs).to(
                    self.device,
                    dtype=self.dtype
                )
            else:
                processor_kwargs = {
                    "hidden_size": hidden_size,
                    "cross_attention_dim": cross_attention_dim,
                    "scale": scale,
                    "num_tokens": self.tokens
                }
                if self.use_face_id:
                    processor_kwargs["rank"] = self.lora_rank
                    if use_torch_2:
                        ip_attn_class = FaceIDLoRAIPAttentionProcessor2_0
                    else:
                        ip_attn_class = FaceIDLoRAIPAttentionProcessor
                elif use_torch_2:
                    ip_attn_class = IPAttentionProcessor2_0
                else:
                    ip_attn_class = IPAttentionProcessor

                new_attention_processors[name] = ip_attn_class(**processor_kwargs).to(
                    self.device,
                    dtype=self.dtype
                )

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
                if isinstance(controlnets[controlnet], SparseControlNetModel):
                    continue
                new_processors: Dict[str, Any] = {}
                current_processors = controlnets[controlnet].attn_processors
                for key in current_processors:
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
        return self.model in ["plus", "plus-face", "full-face"]

    @property
    def use_mlp(self) -> bool:
        """
        Returns true if using a full model
        """
        return self.model == "full-face" and not self.is_sdxl

    @property
    def use_plus_proj(self) -> bool:
        """
        Returns true if using a plus face ID model
        """
        return self.model == "face-id-plus"

    @property
    def use_mlp_id(self) -> bool:
        """
        Returns true if using a non-plus face ID model
        """
        return self.model in ["face-id", "face-id-portrait"]

    @property
    def use_face_id(self) -> bool:
        """
        Returns true if using any face ID model
        """
        return self.model in ["face-id", "face-id-portrait", "face-id-plus"]

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

    def set_scale(self, unet: UNet2DConditionModel, scale: float) -> None:
        """
        Sets the scale on attention processors.
        """
        def get_attn_processors(module: torch.nn.Module) -> Iterator[AttentionProcessor]:
            if hasattr(module, "processor"):
                yield module.processor
            for name, child in module.named_children():
                for processor in get_attn_processors(child):
                    yield processor

        from enfugue.diffusion.support.ip.attention import ( # type: ignore[attr-defined]
            IPAttentionProcessor,
            IPAttentionProcessor2_0,
            LoRAIPAttentionProcessor,
            FaceIDLoRAIPAttentionProcessor,
            FaceIDLoRAIPAttentionProcessor2_0
        )

        for processor in get_attn_processors(unet):
            for attention_class in [
                IPAttentionProcessor,
                IPAttentionProcessor2_0,
                LoRAIPAttentionProcessor,
                FaceIDLoRAIPAttentionProcessor,
                FaceIDLoRAIPAttentionProcessor2_0
            ]:
                if isinstance(processor, attention_class):
                    processor.scale = scale
                    break

        self.scale = scale

    @property
    def face_id_encoder_model(self) -> str:
        """
        Gets the path to the IP model for face ID (1.5 and XL)
        Downloads if needed
        """
        return self.get_model_file(
            self.FACE_ID_ENCODER_PATH,
            directory=self.kwargs.get("clip_vision_dir", None),
            filename="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def face_id_encoder_config(self) -> str:
        """
        Gets the path to the IP model for face ID (1.5 and XL)
        Downloads if needed
        """
        return self.get_model_file(
            self.FACE_ID_ENCODER_CONFIG_PATH,
            directory=self.kwargs.get("clip_vision_dir", None),
            filename="CLIP-ViT-H-14-laion2B-s32B-b79K-config.json"
        )

    @property
    def default_encoder_model(self) -> str:
        """
        Gets the path to the IP model for 1.5
        Downloads if needed
        """
        if self.use_plus_proj:
            return self.face_id_encoder_model
        return self.get_model_file(
            self.DEFAULT_ENCODER_PATH,
            directory=self.kwargs.get("clip_vision_dir", None),
            filename="ip-adapter_sd15_encoder.pth",
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def default_encoder_config(self) -> str:
        """
        Gets the path to the IP model for 1.5
        Downloads if needed
        """
        if self.use_plus_proj:
            return self.face_id_encoder_config
        return self.get_model_file(
            self.DEFAULT_ENCODER_CONFIG_PATH,
            directory=self.kwargs.get("clip_vision_dir", None),
            filename="ip-adapter_sd15_encoder_config.json"
        )

    @property
    def default_image_prompt_checkpoint(self) -> str:
        """
        Gets the path to the IP checkpoint for 1.5
        Downloads if needed
        """
        if self.model == "full-face":
            model_url = self.FACE_FULL_ADAPTER_PATH
            filename = "ip-adapter-full-face_sd15.pth"
        elif self.model == "plus-face":
            model_url = self.FACE_ADAPTER_PATH
            filename = "ip-adapter-plus-face_sd15.pth"
        elif self.model == "plus":
            model_url = self.FINE_GRAINED_ADAPTER_PATH
            filename = "ip-adapter-plus_sd15.pth"
        elif self.model == "face-id":
            model_url = self.FACE_ID_ADAPTER_PATH
            filename = "ip-adapter-faceid_sd15.pth"
        elif self.model == "face-id-plus":
            model_url = self.FACE_ID_PLUS_PATH
            filename = "ip-adapter-faceid-plusv2_sd15.pth"
        elif self.model == "face-id-portrait":
            model_url = self.FACE_ID_PORTRAIT_PATH
            filename = "ip-adapter-faceid-portrait_sd15.pth"
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
        if self.use_plus_proj:
            return self.face_id_encoder_model
        return self.get_model_file(
            self.XL_ENCODER_PATH,
            directory=self.kwargs.get("clip_vision_dir", None),
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
        if self.use_plus_proj:
            return self.face_id_encoder_config
        return self.get_model_file(
            self.XL_ENCODER_CONFIG_PATH,
            directory=self.kwargs.get("clip_vision_dir", None),
            filename="ip-adapter_sdxl_encoder_config.json"
        )

    @property
    def xl_image_prompt_checkpoint(self) -> str:
        """
        Gets the path to the IP checkpoint for XL
        Downloads if needed
        """
        if self.model in ["plus-face", "full-face"]:
            model_url = self.FACE_XL_ADAPTER_PATH
            filename = "ip-adapter-plus-face_sdxl_vit-h.pth"
        elif self.model == "plus":
            model_url = self.FINE_GRAINED_XL_ADAPTER_PATH
            filename = "ip-adapter-plus_sdxl_vit-h.pth"
        elif self.model == "face-id":
            model_url = self.FACE_ID_XL_ADAPTER_PATH
            filename = "ip-adapter-faceid_sdxl.pth"
        elif self.model == "face-id-plus":
            model_url = self.FACE_ID_XL_PLUS_PATH
            filename = "ip-adapter-faceid-plusv2_sdxl.pth"
        else:
            model_url = self.XL_ADAPTER_PATH
            filename = "ip-adapter_sdxl.pth"
            
        return self.get_model_file(
            model_url,
            filename=filename,
            extensions=[".bin", ".pth", ".safetensors"]
        )

    @property
    def default_tokens(self) -> int:
        """
        Gets the default number of tokens based on the model.
        """
        if self.model == "full-face" and not self.is_sdxl:
            return 257
        elif self.model in ["plus", "plus-face", "full-face"] or (
            self.model in ["face-id-portrait"] and not self.is_sdxl
        ):
            return 16
        return 4

    @property
    def combine_conditions(self) -> bool:
        """
        Returns whether or not conditions should be combined
        """
        return self.model == "face-id-portrait" and not self.is_sdxl

    @property
    def tokens(self) -> int:
        """
        Gets the number of tokens for extra clip context
        """
        return getattr(self, "_tokens", self.default_tokens)

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
        from transformers.models.clip import (
            CLIPVisionModelWithProjection,
            CLIPVisionConfig
        )
        if not hasattr(self, "_encoder"):
            if self.is_sdxl:
                logger.debug(f"Initializing CLIPVisionModelWithProjection from {self.xl_encoder_model} using configuration {self.xl_encoder_config}")
                model = self.xl_encoder_model
                config_dict = load_json(self.xl_encoder_config)
            else:
                logger.debug(f"Initializing CLIPVisionModelWithProjection from {self.default_encoder_model} using configuration {self.default_encoder_config}")
                model = self.default_encoder_model
                config_dict = load_json(self.default_encoder_config)
            if "vision_config" in config_dict:
                config_dict = config_dict["vision_config"]
            self._encoder = CLIPVisionModelWithProjection.from_pretrained(
                model,
                config=CLIPVisionConfig.from_dict(config_dict),
                use_safetensors="safetensors" in model,
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
            if self.use_mlp:
                from enfugue.diffusion.support.ip.projection import MLPProjectionModel
                self._projector = MLPProjectionModel(
                    cross_attention_dim=self.cross_attention_dim,
                    clip_embeddings_dim=self.encoder.config.hidden_size,
                )
            elif self.use_mlp_id:
                from enfugue.diffusion.support.ip.projection import FaceIDMLPProjectionModel
                self._projector = FaceIDMLPProjectionModel( # type: ignore[assignment]
                    cross_attention_dim=self.cross_attention_dim,
                    id_embeddings_dim=512,
                    num_tokens=self.tokens,
                )
            elif self.use_plus_proj:
                from enfugue.diffusion.support.ip.projection import ProjectionPlusModel
                self._projector = ProjectionPlusModel( # type: ignore[assignment]
                    cross_attention_dim=self.cross_attention_dim,
                    id_embeddings_dim=512,
                    clip_embeddings_dim=self.encoder.config.hidden_size,
                    num_tokens=self.tokens,
                )
            elif self.use_fine_grained:
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
                self._projector = ImageProjectionModel( # type: ignore[assignment]
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

    @property
    def face_analyzer(self) -> FaceAnalyzer:
        """
        Gets the face analyzer if set. Otherwise, creates.
        """
        if not hasattr(self, "_face_analyzer"):
            from enfugue.diffusion.support.face import FaceAnalyzer
            self._face_analyzer = FaceAnalyzer(
                self.root_dir,
                self.kwargs.get("detection_dir", self.model_dir),
                device=self.device,
                dtype=self.dtype,
                offline=self.offline
            )
            self._face_analyzer.task_callback = self.task_callback
        return self._face_analyzer

    @property
    def face_processor(self) -> FaceAnalyzerImageProcessor:
        """
        Gets the face processor if set.
        """
        if not hasattr(self, "_face_processor"):
            raise RuntimeError("Face processor not initialized before being requested.")
        return self._face_processor

    @face_processor.setter
    def face_processor(self, analyze: FaceAnalyzerImageProcessor) -> None:
        """
        Sets the face processor.
        """
        self._face_processor = analyze

    @face_processor.deleter
    def face_processor(self) -> None:
        """
        Unsets the face processor.
        """
        if hasattr(self, "_face_processor"):
            del self._face_processor

    @contextmanager
    def context(self) -> Iterator[Self]:
        """
        Override parent context to send models to device
        """
        import torch
        with super(IPAdapter, self).context():
            self.projector.to(device=self.device, dtype=self.dtype)
            if self.use_face_id:
                if self.use_plus_proj:
                    self.encoder.to(device=self.device, dtype=self.dtype)
                with self.face_analyzer.insightface() as face_processor:
                    self.face_processor = face_processor
                    with torch.inference_mode():
                        yield self
                    del self.face_processor
            else:
                self.encoder.to(device=self.device, dtype=self.dtype)
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
        if not isinstance(images, list):
            images = [images]

        if self.use_face_id:
            faceid_tensors = [
                self.face_processor(image)
                for image in images
            ]
            faceid_tensors = [
                tensor for tensor in faceid_tensors
                if tensor is not None
            ]

            if faceid_tensors:
                faceid_embeds = torch.cat(faceid_tensors, dim=0) # type: ignore[arg-type]
            else:
                faceid_embeds = torch.zeros((1, self.projector.id_embeddings_dim))

            faceid_embeds = faceid_embeds.to(
                device=self.device,
                dtype=self.dtype
            )
        else:
            faceid_embeds = None

        if not self.use_face_id or self.use_plus_proj:
            clip_image = self.processor(
                images=images,
                return_tensors="pt"
            ).pixel_values

            clip_image = clip_image.to(self.device, dtype=self.dtype)
            clip_image_encoded = self.encoder(
                clip_image,
                output_hidden_states=self.use_fine_grained or self.use_plus_proj
            )

            if self.use_fine_grained or self.use_plus_proj:
                clip_image_embeds = clip_image_encoded.hidden_states[-2]
            else:
                clip_image_embeds = clip_image_encoded.image_embeds
        else:
            clip_image_embeds = None

        if self.use_fine_grained or self.use_plus_proj:
            uncond_clip_image_embeds = self.encoder(
                torch.zeros_like(clip_image),
                output_hidden_states=True
            ).hidden_states[-2]
        elif clip_image_embeds is not None:
            uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)
        else:
            uncond_clip_image_embeds = None

        if clip_image_embeds is not None:
            clip_image_embeds = clip_image_embeds.to(dtype=self.dtype)
        if uncond_clip_image_embeds is not None:
            uncond_clip_image_embeds = uncond_clip_image_embeds.to(dtype=self.dtype)

        if self.use_face_id:
            if self.use_plus_proj:
                image_prompt_embeds = self.projector(faceid_embeds, clip_image_embeds, shortcut=True, scale=self.scale)
                image_uncond_prompt_embeds = self.projector(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=True, scale=self.scale)
            else:
                image_prompt_embeds = self.projector(faceid_embeds)
                image_uncond_prompt_embeds = self.projector(torch.zeros_like(faceid_embeds))
        else:
            image_prompt_embeds = self.projector(clip_image_embeds)
            image_uncond_prompt_embeds = self.projector(uncond_clip_image_embeds)

        return image_prompt_embeds, image_uncond_prompt_embeds
