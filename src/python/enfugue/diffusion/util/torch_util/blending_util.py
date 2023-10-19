# Adapted from https://github.com/WASasquatch/PPF_Noise_ComfyUI/tree/main
from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from enfugue.diffusion.constants import LATENT_BLEND_METHOD_LITERAL

__all__ = ["LatentBlender", "blend_latents"]

class LatentBlender:
    """
    Provides a number of methods of blending latents.
    """
    @classmethod
    def normalize(
        cls,
        latent: Tensor,
        lower: Optional[Union[Tensor, float]] = None,
        upper: Optional[Union[Tensor, float]] = None,
    ) -> Tensor:
        """
        Normalizes a latent between bounds
        """
        minimum = latent.min()
        maximum = latent.max()
        if lower is None:
            lower = minimum
        if upper is None:
            upper = maximum

        normalized = (latent - minimum) / (maximum - minimum)
        return normalized * (upper - lower) + lower

    @classmethod
    def add(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Returns a weighted sum (not ratio)
        """
        return left * time + right * time

    @classmethod
    def bislerp(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Smooth bidirectional lerp
        """
        return cls.normalize((1 - time) * left + time * right)

    @classmethod
    def cosine(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Uses torch.cos
        """
        import torch
        from math import pi
        return (left + right - (left - right) * torch.cos(time * torch.tensor(pi))) / 2

    @classmethod
    def cubic(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Uses a cubic function
        """
        return left + (right - left) * (3 * time ** 2 - 2 * time ** 3)
    
    @classmethod
    def subtract(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Subtracts right from left
        """
        return (left * time - right * time)

    @classmethod
    def difference(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Adds the difference between two tensors
        """
        return cls.normalize(abs(left - right) * time)

    @classmethod
    def exclusion(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Produces a contrasting effect
        """
        return cls.normalize((left + right - 2 * left * right) * time)

    @classmethod
    def glow(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Use the right image to make the left glow
        """
        import torch
        return torch.where(
            left <= 1,
            left ** 2 / (1 - right + 1e-6),
            right * (left - 1) / (left + 1e-6)
        )

    @classmethod
    def inject(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Injects the right into the left with a factor
        """
        return left + (right * time)

    @classmethod
    def sqlerp(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Square root lerp
        """
        sqrt_alpha = time ** 0.5
        sqrt_one_minus_alpha = (1.0 - time) ** 0.5
        return left * sqrt_alpha + right * sqrt_one_minus_alpha

    @classmethod
    def lerp(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Linear interpolation
        """
        return left * (1 - time) + right * time

    @classmethod
    def slerp(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Spherical interpolation
        """
        import torch
        lhs = left * torch.sin((1 - time) * torch.acos(torch.clamp(torch.sum(left * right, dim=1), -1.0, 1.0)))
        rhs = right * torch.sin(time * torch.acos(torch.clamp(torch.sum(left * right, dim=1), -1.0, 1.0)))
        denominator = torch.sin(torch.acos(torch.clamp(torch.sum(left * right, dim=1), -1.0, 1.0)))
        return cls.normalize((lhs + rhs) / denominator)

    @classmethod
    def multiply(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Multiplies tensors together by a factor
        """
        return cls.normalize(left * time * right * time)

    @classmethod
    def overlay(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Combines tensors in a subtler more dramatic way
        """
        import torch
        if torch.all(right < 0.5):
            return (2 * left * right + left**2 - 2 * left * right * left) * time
        return (1 - 2 * (1 - left) * (1 - right)) * time
    
    @classmethod
    def screen(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        High-key blending
        """
        return cls.normalize(1 - (1 - left) * (1 - right) * (1 - time))

    @classmethod
    def linear_dodge(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        High-key brightening
        """
        return cls.normalize(left + right * time)

    @classmethod
    def color_dodge(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Brightens left based on right
        """
        return left / (1 - right + 1e-6)

    @classmethod
    def pin_light(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Preserve details, intensify colors
        """
        import torch
        return torch.where(
            right <= 0.5,
            torch.min(left, 2 * right),
            torch.max(left, 2 * right - 1)
        )
    
    @classmethod
    def vivid_light(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Strongly brightens the left
        """
        import torch
        return torch.where(
            right <= 0.5,
            left / (1 - 2 * right + 1e-6),
            (left + 2 * right - 1) / (2 * (1 - right) + 1e-6)
        )

    @classmethod
    def hard_light(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        Use the right image as a hard light for the left
        """
        return (
            2 * left * right * (left < 0.5).to(left.dtype) +
            (1 - 2 * (1 - left) * (1 - right)) * (left >= 5).to(left.dtype)
        ) * time

    @classmethod
    def linear_light(cls, left: Tensor, right: Tensor, time: Union[Tensor, float]) -> Tensor:
        """
        High-constrast brightening
        """
        import torch
        return torch.where(
            right <= 0.5,
            left + 2 * right - 1,
            left + 2 * (right - 0.5)
        )

def blend_latents(
    left: Tensor,
    right: Tensor,
    time: Union[Tensor, float],
    method: LATENT_BLEND_METHOD_LITERAL = "lerp",
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tensor:
    """
    Calls the appropriate interpolation method
    """
    if left.shape != right.shape:
        raise ValueError(f"Cannot blend latents of different shapes; left is {left.shape}, right is {right.shape}")
    try:
        return {
            "add": LatentBlender.add,
            "bislerp": LatentBlender.bislerp,
            "cosine": LatentBlender.cosine,
            "cubic": LatentBlender.cubic,
            "difference": LatentBlender.difference,
            "inject": LatentBlender.inject,
            "lerp": LatentBlender.lerp,
            "slerp": LatentBlender.slerp,
            "exclusion": LatentBlender.exclusion,
            "subtract": LatentBlender.subtract,
            "multiply": LatentBlender.multiply,
            "overlay": LatentBlender.overlay,
            "screen": LatentBlender.screen,
            "color_dodge": LatentBlender.color_dodge,
            "linear_dodge": LatentBlender.linear_dodge,
            "glow": LatentBlender.glow,
            "pin_light": LatentBlender.pin_light,
            "hard_light": LatentBlender.hard_light,
            "linear_light": LatentBlender.linear_light,
            "vivid_light": LatentBlender.vivid_light,
            "sqlerp": LatentBlender.sqlerp
        }[method](left, right, time)
    except KeyError:
        raise ValueError(f"Unknown interpolation method {method}")
