from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Callable, Any, Literal, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import (
        Tensor,
        Generator,
        device as Device,
        dtype as DType
    )
    from enfugue.diffusion.constants import NOISE_METHOD_LITERAL

__all__ = ["NoiseMaker", "make_noise"]

@dataclass
class NoiseMaker:
    batch_size: int
    channels: int
    height: int
    width: int
    animation_frames: Optional[int] = None
    generator: Optional[Generator] = None
    device: Optional[Device] = None
    dtype: Optional[DType] = None

    @property
    def shape(self) -> Union[
        Tuple[int, int, int, int],
        Tuple[int, int, int, int, int],
    ]:
        """
        Gets the shape of the return tensor
        """
        if self.animation_frames:
            return (
                self.batch_size,
                self.channels,
                self.animation_frames,
                self.height,
                self.width,
             )
        return (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
         )

    def default(self) -> Tensor:
        """
        Uses the default torch.rand
        """
        import torch
        return torch.randn(
            self.shape,
            generator=self.generator,
            dtype=self.dtype,
            layout=torch.strided
        ).to(self.device)

    def power(
        self,
        alpha: float = 1.0,
        scale: float = 1.0,
        modulator: float = 0.1,
        noise_type: Literal[
            "white", "grey", "pink",
            "green", "blue", "violet",
            "velvet", "random_mix", "brownian_fractal"
        ] = "brownian_fractal"
    ) -> Tensor:
        """
        Calculates power law noise
        """
        import torch
        from enfugue.diffusion.util.torch_util.noise_util.power import PowerLawNoise # type: ignore
        frames = 1 if self.animation_frames is None else self.animation_frames
        shape = (
            frames,
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        power_generator = PowerLawNoise()
        for i in range(frames):
            noise[i, :, :, :, 0:min(self.channels, 3)] = power_generator(
                batch_size=self.batch_size,
                width=self.width,
                height=self.height,
                alpha=alpha,
                scale=scale,
                modulator=modulator,
                noise_type=noise_type,
                generator=self.generator,
            )[:, :, :, 0:min(self.channels, 3)]

        from einops import rearrange
        noise = rearrange(noise, "f b h w c -> b c f h w")
        if self.animation_frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    def simplex(self) -> Tensor:
        """
        Calculates simplex noise
        """
        import torch
        import numpy as np
        import opensimplex
        frames = 1 if self.animation_frames is None else self.animation_frames
        shape = (
            frames,
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        opensimplex.seed(int(torch.randint(2**32, (1,), generator=self.generator)[0]))
        for i in range(frames):
            noise[i, :, :, :, :] = torch.from_numpy(
                opensimplex.noise4array(
                    np.arange(self.channels),
                    np.arange(self.width),
                    np.arange(self.height),
                    np.arange(self.batch_size),
                )
            )
        from einops import rearrange
        noise = rearrange(noise, "f b h w c -> b c f h w")
        if self.animation_frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    def crosshatch(
        self,
        frequency: int=320,
        octaves: int=12,
        persistence: float=1.5,
        angle_degrees: int=45,
        brightness: float=0.0,
        contrast: float=0.0,
        blur: int=1,
        color_tolerance: float=0.01,
        num_colors: int=32,
        clamp_min: float=0.0,
        clamp_max: float=1.0,
    ) -> Tensor:
        """
        Calculates crosshatch noise
        """
        import torch
        from enfugue.diffusion.util.torch_util.noise_util.crosshatch import CrossHatchPowerFractal # type: ignore
        frames = 1 if self.animation_frames is None else self.animation_frames
        shape = (
            frames,
            self.batch_size,
            self.height,
            self.width,
            self.channels,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        crosshatch = CrossHatchPowerFractal(
            width=self.width,
            height=self.height,
            frequency=frequency,
            octaves=octaves,
            persistence=persistence,
            num_colors=num_colors,
            color_tolerance=color_tolerance,
            angle_degrees=angle_degrees,
            blur=blur,
            brightness=brightness,
            contrast=contrast,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        for i in range(frames):
            noise[i, :, :, :, 0:min(self.channels, 3)] = crosshatch(
                batch_size=self.batch_size,
                generator=self.generator,
            )[:, :, :, 0:min(self.channels, 3)]

        from einops import rearrange
        noise = rearrange(noise, "f b h w c -> b c f h w")
        if self.animation_frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    def perlin(
        self,
        evolution_factor: float=0.1,
        octaves: int=4,
        persistence: float=0.5,
        lacunarity: float=2.0,
        exponent: float=4.0,
        scale: int=4,
        brightness: float=0.0,
        contrast: float=0.0,
        min_clamp: float=0.0,
        max_clamp: float=1.0,
    ) -> Tensor:
        """
        Calculates perlin noise
        """
        import torch
        from enfugue.diffusion.util.torch_util.noise_util.perlin import PerlinPowerFractal # type: ignore
        frames = 1 if self.animation_frames is None else self.animation_frames
        shape = (
            frames,
            self.channels,
            self.batch_size,
            self.height,
            self.width,
        )
        noise = torch.ones(shape, dtype=torch.float32, device="cpu").cpu()
        perlin = PerlinPowerFractal(self.width, self.height)

        for i in range(frames):
            for j in range(3):
                noise[i, j, :, :, :] = perlin(
                    batch_size=self.batch_size,
                    X=0,
                    Y=0,
                    Z=0,
                    frame=i,
                    evolution_factor=evolution_factor,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    exponent=exponent,
                    scale=scale,
                    brightness=brightness,
                    contrast=contrast,
                    min_clamp=min_clamp,
                    max_clamp=max_clamp
                )[:, :, :, 0]

        from einops import rearrange
        noise = rearrange(noise, "f c b h w -> b c f h w")
        if self.animation_frames is None:
            noise = noise[:, :, 0, :, :]
        return noise.to(self.device, dtype=self.dtype)

    @classmethod
    def get_method_by_name(cls, method: NOISE_METHOD_LITERAL) -> Callable:
        """
        Gets the callable method by name
        """
        if method == "default":
            return cls.default
        elif method == "crosshatch":
            return cls.crosshatch
        elif method == "simplex":
            return cls.simplex
        elif method == "perlin":
            return cls.perlin
        return cls.power

    @classmethod
    def get_method_kwargs(
        cls,
        method: NOISE_METHOD_LITERAL,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Gets keyword arguments for the callable method
        """
        import inspect
        key_names = set(inspect.signature(cls.get_method_by_name(method)).parameters.keys())
        method_kwargs = dict([
            (key, value)
            for key, value in kwargs.items()
            if key in key_names
        ])
        if "noise_type" in key_names:
            method_kwargs["noise_type"] = method
        return method_kwargs

def make_noise(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    animation_frames: Optional[int] = None,
    generator: Optional[Generator] = None,
    device: Optional[Device] = None,
    dtype: Optional[DType] = None,
    method: NOISE_METHOD_LITERAL = "default",
    **kwargs: Any
) -> Tensor:
    """
    Executes the passed method
    """
    noise_maker = NoiseMaker(
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        animation_frames=animation_frames,
        generator=generator,
        device=device,
        dtype=dtype
    )
    make_noise_method = noise_maker.get_method_by_name(method)
    return make_noise_method(
        noise_maker,
        **noise_maker.get_method_kwargs(method, **kwargs)
    )
