from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Union, Dict, Optional, TYPE_CHECKING
from typing_extensions import Self

from enfugue.diffusion.constants import MASK_TYPE_LITERAL

if TYPE_CHECKING:
    from torch import (
        Tensor,
        dtype as DType,
        device as Device
    )

__all__ = ["MaskWeightBuilder"]

@dataclass(frozen=True)
class DiffusionMask:
    """
    Holds all the variables needed to compute a mask.
    """
    width: int
    height: int

    def calculate(self) -> Tensor:
        """
        These weights are always 1.
        """
        import torch
        return torch.ones(self.height, self.width)

@dataclass(frozen=True)
class DiffusionUnmask(DiffusionMask):
    """
    Holds all variables need to compute an unmask.
    Unmasks are used to ensure no area of a diffusion is completely ommitted by chunks.
    There is probably a much more efficient way to calculate this. Help is welcomed!
    """
    left: bool
    top: bool
    right: bool
    bottom: bool

    def unmask_left(self, x: int, y: int, midpoint_x: int, midpoint_y: int) -> bool:
        """
        Determines if the left should be unmasked.
        """
        if not self.left:
            return False
        if x > midpoint_x:
            return False
        if y > midpoint_y:
            return x <= self.height - y
        return x <= y

    def unmask_top(self, x: int, y: int, midpoint_x: int, midpoint_y: int) -> bool:
        """
        Determines if the top should be unmasked.
        """
        if not self.top:
            return False
        if y > midpoint_y:
            return False
        if x > midpoint_x:
            return y <= self.width - x
        return y <= x

    def unmask_right(self, x: int, y: int, midpoint_x: int, midpoint_y: int) -> bool:
        """
        Determines if the right should be unmasked.
        """
        if not self.right:
            return False
        if x < midpoint_x:
            return False
        if y > midpoint_y:
            return x >= y
        return x >= self.height - y

    def unmask_bottom(self, x: int, y: int, midpoint_x: int, midpoint_y: int) -> bool:
        """
        Determines if the bottom should be unmasked.
        """
        if not self.bottom:
            return False
        if y < midpoint_y:
            return False
        if x > midpoint_x:
            return y >= x
        return y >= self.width - x

    def calculate(self) -> Tensor:
        """
        Calculates the unmask.
        """
        import torch

        unfeather_mask = torch.zeros(self.height, self.width)
        midpoint_x = self.width // 2
        midpoint_y = self.height // 2

        for y in range(self.height):
            for x in range(self.width):
                if (
                    self.unmask_left(x, y, midpoint_x, midpoint_y) or
                    self.unmask_top(x, y, midpoint_x, midpoint_y) or
                    self.unmask_right(x, y, midpoint_x, midpoint_y) or
                    self.unmask_bottom(x, y, midpoint_x, midpoint_y)
                ):
                    x_deviation = abs(midpoint_x - x) / self.width
                    y_deviation = abs(midpoint_y - y) / self.height
                    unfeather_mask[y, x] = min(1.0, 1.0 * max(x_deviation, y_deviation) / 0.29)

        return unfeather_mask

@dataclass(frozen=True)
class BilinearDiffusionMask(DiffusionMask):
    """
    Feathers the edges of a mask.
    """
    ratio: float

    def calculate(self) -> Tensor:
        """
        Calculates weights in linear gradients.
        """
        import torch
        tensor = super(BilinearDiffusionMask, self).calculate()
        latent_length = int(self.ratio * self.width)
        for i in range(latent_length):
            feathered = torch.tensor(i / latent_length)
            tensor[:, i] = torch.minimum(tensor[:, i], feathered)
            tensor[i, :] = torch.minimum(tensor[i, :], feathered)
            tensor[:, self.width - i - 1] = torch.minimum(
                tensor[:, self.width - i - 1],
                feathered
            )
            tensor[self.height - i - 1, :] = torch.minimum(
                tensor[self.height - i - 1, :],
                feathered
            )
        return tensor

@dataclass(frozen=True)
class GaussianDiffusionMask(DiffusionMask):
    """
    Feathers the edges and corners using gaussian probability.
    """
    deviation: float
    growth: float

    def calculate(self) -> Tensor:
        """
        Calculates weights with a gaussian distribution
        """
        import torch
        import numpy as np
        midpoint = (self.width - 1) / 2
        x_probabilities = [
            np.exp(-(x - midpoint) * (x - midpoint) / (self.width ** (2 + self.growth)) / (2 * self.deviation)) / np.sqrt(2 * np.pi * self.deviation)
            for x in range(self.width)
        ]
        midpoint = (self.height - 1) / 2
        y_probabilities = [
            np.exp(-(y - midpoint) * (y - midpoint) / (self.height ** (2 + self.growth)) / (2 * self.deviation)) / np.sqrt(2 * np.pi * self.deviation)
            for y in range(self.height)
        ]

        weights = np.outer(y_probabilities, x_probabilities)
        weights = torch.tile(torch.tensor(weights), (1, 1)) # type: ignore[assignment]
        return weights # type: ignore[return-value]

@dataclass
class MaskWeightBuilder:
    """
    A class for computing blending masks given dimensions and some optional parameters

    Stores masks on the device for speed. Be sure to free memory when no longer needed.
    """
    device: Union[str, Device]
    dtype: DType

    constant_weights: Dict[DiffusionMask, Tensor] = field(default_factory=dict)
    unmasked_weights: Dict[DiffusionUnmask, Tensor] = field(default_factory=dict)
    bilinear_weights: Dict[BilinearDiffusionMask, Tensor] = field(default_factory=dict)
    gaussian_weights: Dict[GaussianDiffusionMask, Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        """
        Clears all stored tensors
        """
        for key in list(self.constant_weights.keys()):
            del self.constant_weights[key]
        for key in list(self.bilinear_weights.keys()):
            del self.bilinear_weights[key]
        for key in list(self.gaussian_weights.keys()):
            del self.gaussian_weights[key]
        for key in list(self.unmasked_weights.keys()):
            del self.unmasked_weights[key]

    def __enter__(self) -> Self:
        """
        Implement base enter.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        On exit, clear tensors.
        """
        self.clear()

    def constant(
        self,
        width: int,
        height: int,
        **kwargs: Any
    ) -> Tensor:
        """
        Calculates the constant mask. No feathering.
        """
        mask = DiffusionMask(
            width=width,
            height=height
        )
        if mask not in self.constant_weights:
            self.constant_weights[mask] = mask.calculate().to(
                dtype=self.dtype,
                device=self.device
            )
        return self.constant_weights[mask]

    def bilinear(
        self,
        width: int,
        height: int,
        unfeather_left: bool = False,
        unfeather_top: bool = False,
        unfeather_right: bool = False,
        unfeather_bottom: bool = False,
        ratio: float = 0.25,
        **kwargs: Any
    ) -> Tensor:
        """
        Calculates the bilinear mask.
        """
        import torch
        mask = BilinearDiffusionMask(
            width=width,
            height=height,
            ratio=ratio
        )
        unmask = DiffusionUnmask(
            width=width,
            height=height,
            left=unfeather_left,
            top=unfeather_top,
            right=unfeather_right,
            bottom=unfeather_bottom
        )
        if mask not in self.bilinear_weights:
            self.bilinear_weights[mask] = mask.calculate().to(
                dtype=self.dtype,
                device=self.device
            )
        if unmask not in self.unmasked_weights:
            self.unmasked_weights[unmask] = unmask.calculate().to(
                dtype=self.dtype,
                device=self.device
            )
        return torch.maximum(
            self.bilinear_weights[mask],
            self.unmasked_weights[unmask]
        )

    def gaussian(
        self,
        width: int,
        height: int,
        unfeather_left: bool = False,
        unfeather_top: bool = False,
        unfeather_right: bool = False,
        unfeather_bottom: bool = False,
        deviation: float = 0.01,
        growth: float = 0.15,
        **kwargs: Any
    ) -> Tensor:
        """
        Calculates the gaussian mask, optionally unfeathered.
        """
        import torch
        mask = GaussianDiffusionMask(
            width=width,
            height=height,
            deviation=deviation,
            growth=growth
        )
        unmask = DiffusionUnmask(
            width=width,
            height=height,
            left=unfeather_left,
            top=unfeather_top,
            right=unfeather_right,
            bottom=unfeather_bottom
        )
        if mask not in self.gaussian_weights:
            self.gaussian_weights[mask] = mask.calculate().to(
                dtype=self.dtype,
                device=self.device
            )
        if unmask not in self.unmasked_weights:
            self.unmasked_weights[unmask] = unmask.calculate().to(
                dtype=self.dtype,
                device=self.device
            )
        return torch.maximum(
            self.gaussian_weights[mask],
            self.unmasked_weights[unmask]
        )

    def temporal(
        self,
        tensor: Tensor,
        frames: Optional[int] = None,
        unfeather_start: bool = False,
        unfeather_end: bool = False
    ) -> Tensor:
        """
        Potentially expands a tensor temporally
        """
        import torch
        if frames is None:
            return tensor
        tensor = tensor.unsqueeze(2).repeat(1, 1, frames, 1, 1)
        if not unfeather_start or not unfeather_end:
            frame_length = frames // 3
            for i in range(frame_length):
                feathered = torch.tensor(i / frame_length)
                if not unfeather_start:
                    tensor[:, :, i, :, :] = torch.minimum(
                        tensor[:, :, i, :, :],
                        feathered
                    )
                if not unfeather_end:
                    tensor[:, :, frames - i - 1, :, :] = torch.minimum(
                        tensor[:, :, frames - i - 1, :, :],
                        feathered
                    )
        return tensor

    def __call__(
        self,
        mask_type: MASK_TYPE_LITERAL,
        batch: int,
        dim: int,
        width: int,
        height: int,
        frames: Optional[int] = None,
        unfeather_left: bool = False,
        unfeather_top: bool = False,
        unfeather_right: bool = False,
        unfeather_bottom: bool = False,
        unfeather_start: bool = False,
        unfeather_end: bool = False,
        **kwargs: Any
    ) -> Tensor:
        """
        Calculates a mask depending on the method requested.
        """
        if mask_type == "constant":
            get_mask = self.constant
        elif mask_type == "bilinear":
            get_mask = self.bilinear
        elif mask_type == "gaussian":
            get_mask = self.gaussian
        else:
            raise AttributeError(f"Unknown mask type {mask_type}")

        mask = get_mask(
            width=width,
            height=height,
            unfeather_left=unfeather_left,
            unfeather_top=unfeather_top,
            unfeather_right=unfeather_right,
            unfeather_bottom=unfeather_bottom,
            **kwargs
        )

        return self.temporal(
            mask.unsqueeze(0).unsqueeze(0).repeat(batch, dim, 1, 1),
            frames=frames,
            unfeather_start=unfeather_start,
            unfeather_end=unfeather_end,
        )
