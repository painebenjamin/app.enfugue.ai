from __future__ import annotations

from typing import Any, Iterator, Literal, Optional, TYPE_CHECKING

from contextlib import contextmanager

from PIL import Image

from enfugue.diffusion.util import ComputerVision
from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

if TYPE_CHECKING:
    from realesrgan import RealESRGANer
    from enfugue.diffusion.support.upscale.gfpganer import GFPGANer  # type: ignore[attr-defined]
    from enfugue.diffusion.support.upscale.ccsr.model.ccsr_stage2 import ControlLDM  # type: ignore[attr-defined]

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

class CCSRProcessor(SupportModelImageProcessor):
    """
    Holds a reference to the CCSR LDM and provides a callable
    """
    def __init__(self, ccsr: ControlLDM, **kwargs: Any) -> None:
        super(CCSRProcessor, self).__init__(**kwargs)
        self.ccsr = ccsr

    def __call__(
        self,
        image: Image.Image,
        num_steps: int=45,
        strength: float=1.0,
        tile_diffusion_size: Optional[int]=512,
        tile_diffusion_stride: Optional[int]=256,
        tile_vae_decode_size: Optional[int]=160,
        tile_vae_encode_size: Optional[int]=1024,
        color_fix_type: Optional[Literal["wavelet", "adain"]]="adain",
        t_min: float=0.3333,
        t_max: float=0.6667,
        seed: Optional[int]=None,
        positive_prompt: str="",
        negative_prompt: str="",
        cfg_scale: float=1.0,
        outscale: int=2
    ) -> Image.Image:
        """
        Upscales an image using CCSR
        """
        import torch
        import numpy as np
        from einops import rearrange
        from math import ceil
        from enfugue.diffusion.support.upscale.ccsr.utils.image import auto_resize # type: ignore
        from enfugue.diffusion.support.upscale.ccsr.model.q_sampler import SpacedSampler # type: ignore
        # seed
        if seed is not None:
            import pytorch_lightning as pl
            pl.seed_everything(seed)

        # Resize bicubic first
        image = image.convert("RGB").resize(
            tuple(ceil(x*outscale) for x in image.size),
            Image.BICUBIC
        )
        # Get the condition
        condition = auto_resize(image, 512 if not tile_diffusion_size else tile_diffusion_size)
        condition = condition.resize(
            tuple((x//64+1)*64 for x in condition.size),
            Image.LANCZOS
        )
        condition = torch.tensor(
            np.array(condition) / 255.0,
            dtype=torch.float32,
            device=self.ccsr.device
        ).unsqueeze(0)
        condition = rearrange(condition.clamp_(0, 1), "n h w c -> n c h w").contiguous()
        # Create noise
        height, width = condition.shape[-2:]
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=self.ccsr.device, dtype=torch.float32)
        # Adjust control scale in model
        self.ccsr.control_scales = [strength] * 13
        # Instantiate sampler
        sampler = SpacedSampler(self.ccsr, var_type="fixed_small")
        # Set tiling
        if tile_vae_decode_size and tile_vae_encode_size:
            self.ccsr._init_tiled_vae(
                encoder_tile_size=tile_vae_encode_size,
                decoder_tile_size=tile_vae_decode_size
            )
        # Sample
        sample_kwargs = {
            "steps": num_steps,
            "t_max": t_max,
            "t_min": t_min,
            "shape": shape,
            "cond_img": condition,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "x_T": x_T,
            "cfg_scale": cfg_scale,
            "color_fix_type": "none" if not color_fix_type else color_fix_type
        }
        if tile_diffusion_size and tile_diffusion_stride:
            samples = sampler.sample_with_tile_ccsr(
                tile_size=tile_diffusion_size,
                tile_stride=tile_diffusion_stride,
                **sample_kwargs
            )
        else:
            samples = sampler.sample_ccsr(**sample_kwargs)
        # Return to image
        samples = samples.clamp(0, 1)
        samples = (rearrange(samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        return Image.fromarray(samples[0])

class FaceRestoreProcessor(SupportModelImageProcessor):
    """
    Holds a reference to the gfpganer and provides a callable
    """
    def __init__(self, gfpganer: GFPGANer, **kwargs: Any) -> None:
        super(FaceRestoreProcessor, self).__init__(**kwargs)
        self.gfpganer = gfpganer

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Upscales an image
        """
        return ComputerVision.revert_image(
            self.gfpganer.enhance(
                ComputerVision.convert_image(image),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )[2]
        )

class Upscaler(SupportModel):
    """
    The upscaler user ESRGAN or GFGPGAN for up to 4x upscale
    """

    ESRGAN_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ESRGAN_ANIME_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    
    GFPGAN_PATH = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    GFPGAN_DETECTION_PATH = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    GFPGAN_PARSENET_PATH = "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

    CCSR_CKPT_PATH = "https://huggingface.co/benjamin-paine/ccsr/resolve/main/real-world_ccsr.ckpt"

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
            model_path = self.get_model_file(self.ESRGAN_ANIME_PATH)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = self.get_model_file(self.ESRGAN_PATH)

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
    def face_restore(self) -> Iterator[SupportModelImageProcessor]:
        """
        Only does face enhancement
        """
        with self.context():
            from enfugue.diffusion.support.upscale.gfpgan import GFPGANer  # type: ignore[attr-defined]
            model_path = self.get_model_file(self.GFPGAN_PATH)
            detection_model_path = self.get_model_file(self.GFPGAN_DETECTION_PATH)
            parse_model_path = self.get_model_file(self.GFPGAN_PARSENET_PATH)

            gfpganer = GFPGANer(
                model_path=model_path,
                detection_model_path=detection_model_path,
                parse_model_path=parse_model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                device=self.device,
                bg_upsampler=None
            )

            processor = FaceRestoreProcessor(gfpganer)
            yield processor
            del processor
            del gfpganer

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
            from enfugue.diffusion.support.upscale.gfpgan import GFPGANer  # type: ignore[attr-defined]
            model_path = self.get_model_file(self.GFPGAN_PATH)
            detection_model_path = self.get_model_file(self.GFPGAN_DETECTION_PATH)
            parse_model_path = self.get_model_file(self.GFPGAN_PARSENET_PATH)

            gfpganer = GFPGANer(
                model_path=model_path,
                detection_model_path=detection_model_path,
                parse_model_path=parse_model_path,
                upscale=4,
                arch="clean",
                channel_multiplier=2,
                device=self.device,
                bg_upsampler=self.get_upsampler(
                    tile=tile,
                    tile_pad=tile_pad,
                    pre_pad=pre_pad
                )
            )

            processor = GFPGANProcessor(gfpganer)
            yield processor
            del processor
            del gfpganer

    @contextmanager
    def ccsr(self) -> Iterator[SupportModelImageProcessor]:
        """
        Does an upscale using CCSR (content consistent super-resolution)
        """
        with self.context():
            from enfugue.diffusion.support.upscale.ccsr.stage_2 import get_model # type: ignore
            from enfugue.diffusion.util.torch_util import load_state_dict
            ccsr_ckpt_path = self.get_model_file(self.CCSR_CKPT_PATH)
            state_dict = load_state_dict(ccsr_ckpt_path)
            ccsr_model = get_model(state_dict)
            del state_dict
            ccsr_model.freeze()
            ccsr_model.to(self.device)
            processor = CCSRProcessor(ccsr_model)
            yield processor
            del processor
            del ccsr_model

    def __call__(
        self,
        method: Literal["esrgan", "esrganime", "gfpgan", "ccsr"],
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
        elif method == "ccsr":
            context = self.ccsr # type: ignore
        else:
            raise ValueError(f"Unknown upscale method {method}") # type: ignore[unreachable]

        with context(**kwargs) as processor:
            return processor(image, outscale=outscale) # type: ignore
