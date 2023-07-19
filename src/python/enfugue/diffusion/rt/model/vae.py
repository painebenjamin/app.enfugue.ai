import torch
from typing import Tuple, Dict, List

from enfugue.diffusion.rt.model.base import BaseModel


class VAE(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        max_batch_size: int,
        embedding_dim: int,
        use_fp16: bool,
    ):
        super(VAE, self).__init__(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "VAEDecoder"
        self.use_fp16 = use_fp16

    def get_model_key(self) -> str:
        return "vae"

    def get_input_names(self) -> List[str]:
        return ["latent"]

    def get_output_names(self) -> List[str]:
        return ["images"]

    def get_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
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
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size: int, image_height: int, image_width: int) -> Dict[str, Tuple[int, ...]]:
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size: int, image_height: int, image_width: int) -> torch.Tensor:
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )
