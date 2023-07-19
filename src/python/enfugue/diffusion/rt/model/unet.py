import torch
from enfugue.diffusion.rt.model.base import BaseModel
from typing import Dict, List, Tuple


class UNet(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
        use_fp16: bool = True,
        device: str = "cuda",
        max_batch_size: int = 16,
        embedding_dim: int = 768,
        text_maxlen: int = 77,
        unet_dim: int = 4,
    ) -> None:
        super(UNet, self).__init__(
            model=model,
            use_fp16=use_fp16,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_model_key(self) -> str:
        return "unet"

    def get_input_names(self) -> List[str]:
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self) -> List[str]:
        return ["latent"]

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> Dict[str, List[Tuple[int, ...]]]:
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            (min_batch, max_batch),
            _,
            _,
            (min_latent_height, max_latent_height),
            (min_latent_width, max_latent_width),
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "sample": [
                (2 * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (2 * batch_size, self.unet_dim, latent_height, latent_width),
                (2 * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (2 * min_batch, self.text_maxlen, self.embedding_dim),
                (2 * batch_size, self.text_maxlen, self.embedding_dim),
                (2 * max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

    def get_shape_dict(self, batch_size: int, image_height: int, image_width: int) -> Dict[str, Tuple[int, ...]]:
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "encoder_hidden_states": (
                2 * batch_size,
                self.text_maxlen,
                self.embedding_dim,
            ),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> Tuple[torch.Tensor, ...]:
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        return (
            torch.randn(
                2 * batch_size,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(
                2 * batch_size,
                self.text_maxlen,
                self.embedding_dim,
                dtype=dtype,
                device=self.device,
            ),
        )
