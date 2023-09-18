from __future__ import annotations

from typing import Any, Iterator, Literal, TYPE_CHECKING

from contextlib import contextmanager

from PIL import Image

from enfugue.util import check_download_to_dir
from enfugue.diffusion.util import ComputerVision
from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

if TYPE_CHECKING:
    from realesrgan import RealESRGANer
    from enfugue.diffusion.support.upscale.gfpganer import GFPGANer  # type: ignore[attr-defined]

__all__ = ["Upscaler"]

class ESRGANProcessor(SupportModelImageProcessor):
    """
    Holds a reference to the esrganer and provides a callable
    """
    def __init__(self, esrganer: RealESRGANer, **kwargs: Any) -> None:
        super(ESRGANProcessor, self).__init__(**kwargs)
        self.esrganer = esrganer

    def __call__(self, image: Image.Image, outscale: int = 2) -> Image.Image:
        """
        Upscales an image
        """
        return ComputerVision.revert_image(
            self.esrganer.enhance(
                ComputerVision.convert_image(image),
                outscale=outscale
            )[0]
        )

class GFPGANProcessor(SupportModelImageProcessor):
    """
    Holds a reference to the gfpganer and provides a callable
    """
    def __init__(self, gfpganer: GFPGANer, **kwargs: Any) -> None:
        super(GFPGANProcessor, self).__init__(**kwargs)
        self.gfpganer = gfpganer

    def __call__(self, image: Image.Image, outscale: int = 2) -> Image.Image:
        """
        Upscales an image
        GFPGan is fixed at x4 so this fixes the scale here
        """
        result = ComputerVision.revert_image(
            self.gfpganer.enhance(
                ComputerVision.convert_image(image),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )[2]
        )
        width, height = result.size
        multiplier = outscale / 4
        return result.resize((int(width * multiplier), int(height * multiplier)))

class Upscaler(SupportModel):
    """
    The upscaler user ESRGAN or GFGPGAN for up to 4x upscale
    """

    GFPGAN_PATH = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    ESRGAN_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ESRGAN_ANIME_PATH = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    )

    @property
    def esrgan_weights_path(self) -> str:
        """
        Gets the path to the ESRGAN model weights.
        """
        return check_download_to_dir(self.ESRGAN_PATH, self.model_dir)

    @property
    def esrgan_anime_weights_path(self) -> str:
        """
        Gets the path to the ESRGAN anime model weights.
        """
        return check_download_to_dir(self.ESRGAN_ANIME_PATH, self.model_dir)

    @property
    def gfpgan_weights_path(self) -> str:
        """
        Gets the path to the GFPGAN model weights.
        """
        return check_download_to_dir(self.GFPGAN_PATH, self.model_dir)

    def get_upsampler(
        self,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        anime: bool = False
    ) -> RealESRGANer:
        """
        Gets the appropriate upsampler
        """
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if anime:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = self.esrgan_anime_weights_path
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = self.esrgan_weights_path

        return RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            device=self.device,
            half=self.dtype is torch.float16,
        )

    @contextmanager
    def esrgan(
        self,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        anime: bool = False,
    ) -> Iterator[SupportModelImageProcessor]:
        """
        Does a simple upscale
        """
        with self.context():
            esrganer = self.get_upsampler(
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                anime=anime
            )
            processor = ESRGANProcessor(esrganer)
            yield processor
            del processor
            del esrganer

    @contextmanager
    def gfpgan(
        self,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
    ) -> Iterator[SupportModelImageProcessor]:
        """
        Does an upscale with face enhancement
        """
        with self.context():
            from enfugue.diffusion.support.upscale.gfpganer import GFPGANer  # type: ignore[attr-defined]

            gfpganer = GFPGANer(
                model_path=self.gfpgan_weights_path,
                upscale=4,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.get_upsampler(
                    tile=tile,
                    tile_pad=tile_pad,
                    pre_pad=pre_pad
                ),
                rootpath=self.model_dir,
                device=self.device,
            )

            processor = GFPGANProcessor(gfpganer)
            yield processor
            del processor
            del gfpganer

    def __call__(
        self,
        method: Literal["esrgan", "esrganime", "gfpgan"],
        image: Image.Image,
        outscale: int = 2,
        **kwargs: Any
    ) -> Image:
        """
        Performs one quick upscale
        """
        if method == "esrgan":
            context = self.esrgan
        elif method == "esrganime":
            context = self.esrgan # type: ignore
            kwargs["anime"] = True
        elif method == "gfpgan":
            context = self.gfpgan # type: ignore
        else:
            raise ValueError(f"Unknown upscale method {method}") # type: ignore[unreachable]

        with context(**kwargs) as processor:
            return processor(image, outscale=outscale) # type: ignore
