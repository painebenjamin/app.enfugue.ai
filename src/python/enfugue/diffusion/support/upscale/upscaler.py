from __future__ import annotations

import PIL

from typing import Union, TYPE_CHECKING

from enfugue.util import check_download_to_dir
from enfugue.diffusion.support.vision import ComputerVision
from enfugue.diffusion.support.model import SupportModel

if TYPE_CHECKING:
    from realesrgan import RealESRGANer


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

    def get_upsampler(self, tile: int = 0, tile_pad: int = 10, pre_pad: int = 10, anime: bool = False) -> RealESRGANer:
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

    def esrgan(
        self,
        image: Union[str, PIL.Image.Image],
        outscale: int = 4,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        anime: bool = False,
    ) -> PIL.Image.Image:
        """
        Does a simple upscale
        """
        with self.context():
            if type(image) is str:
                image = PIL.Image.open(image)

            esrganer = self.get_upsampler(tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, anime=anime)
            result = ComputerVision.revert_image(
                esrganer.enhance(ComputerVision.convert_image(image), outscale=outscale)[0]
            )
            del esrganer
            return result

    def gfpgan(
        self,
        image: Union[str, PIL.Image.Image],
        outscale: int = 4,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
    ) -> PIL.Image.Image:
        """
        Does an upscale with face enhancement
        """
        with self.context():
            if type(image) is str:
                image = PIL.Image.open(image)

            from enfugue.diffusion.support.upscale.gfpganer import GFPGANer  # type: ignore[attr-defined]

            gfpganer = GFPGANer(
                model_path=self.gfpgan_weights_path,
                upscale=outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.get_upsampler(tile=tile, tile_pad=tile_pad, pre_pad=pre_pad),
                rootpath=self.model_dir,
                device=self.device,
            )

            result = ComputerVision.revert_image(
                gfpganer.enhance(
                    ComputerVision.convert_image(image),
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )[2]
            )
            del gfpganer
            return result
