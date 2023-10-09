from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

@dataclass
class LatentUpscaler:
    """
    Upscales an image in latent space
    """
    mode: Literal["nearest-exact", "bilinear", "bicubic"]
    antialias: bool

    def __call__(self, arg: Tensor, scale: float) -> Tensor:
        """
        Execute the actual interpolation
        """
        import torch.nn.functional as F
        return F.interpolate(
            arg,
            scale_factor=scale,
            mode=self.mode,
            antialias=self.mode in ["bicubic", "bilinear"] and self.antialias
        )

@dataclass
class LatentDownscaler:
    """
    Downscales an image in latent space
    """
    mode: Literal["nearest-exact", "bilinear", "bicubic", "area", "pool-max", "pool-avg"]
    antialias: bool

    @property
    def interpolate_mode(self) -> Optional[str]:
        """
        Retrieve the interpolate mode to pass to F.interpolate
        """
        if self.mode in ["nearest-exact", "bilinear", "bicubic", "area"]:
            return self.mode
        return None

    def interpolate(self, arg: Tensor, **kwargs: Any) -> Tensor:
        """
        Perform the actual interpolation based on config
        """
        import torch.nn.functional as F
        if self.interpolate_mode is not None:
            return F.interpolate(arg, **kwargs)
        elif self.mode == "pool-max":
            return F.max_pool2d(arg, **kwargs)
        elif self.mode == "pool-avg":
            return F.avg_pool2d(arg, **kwargs)
        else:
            raise ValueError(f"Unknonwn interpolation mode {self.mode}")

    def __call__(self, arg: Tensor, scale: float) -> Tensor:
        """
        Determine what interpolation method to use and then use it
        """
        if scale > 1:
            scale = 1.0 / float(scale)
        interpolate_mode = self.interpolate_mode
        if interpolate_mode:
            kwargs = {
                "scale_factor": scale,
                "mode": interpolate_mode,
                "antialias": interpolate_mode in ["bicubic", "bilinear"] and self.antialias
            }
        else:
            kwargs = {
                "kernel_size": int(1.0 / scale)
            }
        return self.interpolate(arg, **kwargs)

@dataclass
class LatentScaler:
    """
    Combines the upscaler and downscaler
    """
    upscale_mode: Literal["nearest-exact", "bilinear", "bicubic"] = "nearest-exact"
    upscale_antialias: bool = False
    downscale_mode: Literal["nearest-exact", "bilinear", "bicubic", "area", "pool-max", "pool-avg"] = "nearest-exact"
    downscale_antialias: bool = False

    @property
    def upscaler(self) -> LatentUpscaler:
        """
        Instantiates the upscaler
        """
        return LatentUpscaler(
            mode=self.upscale_mode,
            antialias=self.upscale_antialias
        )

    @property
    def downscaler(self) -> LatentDownscaler:
        """
        Instantiates the downscaler
        """
        return LatentDownscaler(
            mode=self.downscale_mode,
            antialias=self.downscale_antialias
        )

    def __call__(self, arg: Tensor, scale: float) -> Tensor:
        """
        Upscale or downscale based on scale > or < 1.0
        """
        if scale == 1.0:
            return arg
        elif scale < 1.0:
            return self.downscaler(arg, scale)
        return self.upscaler(arg, scale)
