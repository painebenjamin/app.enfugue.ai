from typing import Dict, Type, Union, Any, Optional, Tuple
from typing_extensions import Self

from dataclasses import dataclass, field

import torch
import numpy as np

from numpy import pi, exp, sqrt

from enfugue.diffusion.constants import MASK_TYPE_LITERAL

__all__ = [
    "cuda_available",
    "tensorrt_available",
    "mps_available",
    "directml_available",
    "get_optimal_device",
    "load_ckpt_state_dict",
    "load_safetensor_state_dict",
    "load_state_dict",
    "get_ram_info",
    "get_vram_info",
    "interpolate_tensors",
    "interpolate_tensors_linear",
    "interpolate_tensors_spherical",
    "DTypeConverter",
    "MaskWeightBuilder"
]

def tensorrt_available() -> bool:
    """
    Returns true if TensorRT is available.
    """
    try:
        import tensorrt
        tensorrt  # silence importchecker
        return True
    except:
        return False

def cuda_available() -> bool:
    """
    Returns true if CUDA is available.
    """
    return torch.cuda.is_available() and torch.backends.cuda.is_built()

def mps_available() -> bool:
    """
    Returns true if MPS is available.
    """
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

def directml_available() -> bool:
    """
    Returns true if directml is available.
    """
    try:
        import torch_directml
        return True
    except:
        return False

def get_optimal_device() -> torch.device:
    """
    Gets the optimal device based on availability.
    """
    if cuda_available():
        return torch.device("cuda")
    elif directml_available():
        import torch_directml
        return torch_directml.device()
    elif mps_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_ram_info() -> Tuple[int, int]:
    """
    Returns RAM amount in bytes as [free, total]
    """
    import psutil
    mem = psutil.virtual_memory()
    return (mem.free, mem.total)

def get_vram_info() -> Tuple[int, int]:
    """
    Returns VRAM amount in bytes as [free, total]
    If no GPU is found, returns RAM info.
    """
    if not cuda_available():
        return get_ram_info()
    return torch.cuda.mem_get_info()

def load_ckpt_state_dict(path: str) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Loads a state dictionary from a .ckpt (old-style) file
    """
    return torch.load(path, map_location="cpu")

def load_safetensor_state_dict(path: str) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Loads a state dictionary from a .safetensor(s) (new-style) file
    """
    from safetensors import safe_open

    checkpoint = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
    return checkpoint

def load_state_dict(path: str) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Loads a state dictionary from file.
    Tries to correct issues with incorrrect formats.
    """
    load_order = [load_safetensor_state_dict, load_ckpt_state_dict]
    if "safetensor" not in path:
        load_order = [load_ckpt_state_dict, load_safetensor_state_dict]

    first_error: Optional[Exception] = None

    for i, loader in enumerate(load_order):
        try:
            return loader(path)
        except Exception as ex:
            if first_error is None:
                first_error = ex

    if first_error is not None:
        raise IOError(f"Recevied exception reading checkpoint {path}, please ensure file integrity.\n{type(first_error).__name__}: {first_error}")
    raise IOError(f"No data read from path {path}")

class DTypeConverter:
    """
    This class converts between numpy and torch types.
    """

    numpy_to_torch: Dict[Type, torch.dtype] = {
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.bool_: torch.bool,
    }

    torch_to_numpy: Dict[torch.dtype, Type] = {
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
        torch.bool: np.bool_,
    }

    @staticmethod
    def from_torch(torch_type: torch.dtype) -> Type:
        """
        Gets the numpy type from torch.
        :raises: KeyError When type is unknown.
        """
        return DTypeConverter.torch_to_numpy[torch_type]

    @staticmethod
    def from_numpy(numpy_type: Type) -> torch.dtype:
        """
        Gets the torch type from nump.
        :raises: KeyError When type is unknown.
        """
        return DTypeConverter.numpy_to_torch[numpy_type]

    @staticmethod
    def __call__(type_to_convert: Union[torch.dtype, Type]) -> Union[torch.dtype, Type]:
        """
        Converts from one type to the other, inferring based on the type passed.
        """
        if isinstance(type_to_convert, torch.dtype):
            return DTypeConverter.from_torch(type_to_convert)
        return DTypeConverter.from_numpy(type_to_convert)

@dataclass(frozen=True)
class DiffusionMask:
    """
    Holds all the variables needed to compute a mask.
    """
    dim: int
    batch: int
    width: int
    height: int
    frames: Optional[int] = None
    unfeather_left: bool = False
    unfeather_top: bool = False
    unfeather_right: bool = False
    unfeather_bottom: bool = False
    unfeather_start: bool = False
    unfeather_end: bool = False

    def unfeather(self, tensor: torch.Tensor, feather_ratio: float) -> torch.Tensor:
        """
        Unfeathers the edges of a tensor if requested.
        This ensures the edges of images are not blurred.
        """
        feather_length = int(feather_ratio * self.width)
        for i in range(feather_length):
            unfeathered = torch.tensor((feather_length - i) / feather_length).to(dtype=tensor.dtype, device=tensor.device)
            if self.unfeather_left:
                if self.frames is None:
                    tensor[:, :, :, i] = torch.maximum(tensor[:, :, :, i], unfeathered)
                else:
                    tensor[:, :, :, :, i] = torch.maximum(tensor[:, :, :, :, i], unfeathered)
            if self.unfeather_top:
                if self.frames is None:
                    tensor[:, :, i, :] = torch.maximum(tensor[:, :, i, :], unfeathered)
                else:
                    tensor[:, :, :, i, :] = torch.maximum(tensor[:, :, :, i, :], unfeathered)
            if self.unfeather_right:
                if self.frames is None:
                    tensor[:, :, :, self.width - i - 1] = torch.maximum(
                        tensor[:, :, :, self.width - i - 1],
                        unfeathered
                    )
                else:
                    tensor[:, :, :, :, self.width - i - 1] = torch.maximum(
                        tensor[:, :, :, :, self.width - i - 1],
                        unfeathered
                    )
            if self.unfeather_bottom:
                if self.frames is None:
                    tensor[:, :, self.height - i - 1, :] = torch.maximum(
                        tensor[:, :, self.height - i - 1, :],
                        unfeathered
                    )
                else:
                   tensor[:, :, :, self.height - i - 1, :] = torch.maximum(
                        tensor[:, :, :, self.height - i - 1, :],
                        unfeathered
                    )

        if self.frames is not None and (self.unfeather_start or self.unfeather_end):
            frame_feather_length = int(feather_ratio * self.frames)
            for i in range(frame_feather_length):
                if self.unfeather_start:
                    tensor[:, :, i, :, :] = torch.maximum(
                        tensor[:, :, i, :, :],
                        unfeathered
                    )
                if self.unfeather_end:
                    tensor[:, :, self.frames - i - 1, :, :] = torch.maximum(
                        tensor[:, :, self.frames - i - 1, :, :],
                        unfeathered
                    )

        return tensor

    def calculate(self, feather_ratio: float = 0.125) -> torch.Tensor:
        """
        These weights are always 1.
        """
        if self.frames is None:
            return torch.ones(self.batch, self.dim, self.height, self.width)
        return torch.ones(self.batch, self.dim, self.frames, self.height, self.width)

@dataclass(frozen=True)
class MultilinearDiffusionMask(DiffusionMask):
    """
    Feathers the edges of a mask. Uses the reverse of the 'unfeather' formula.
    """
    ratio: float = 1 / 8

    def calculate(self, feather_ratio: float = 0.125) -> torch.Tensor:
        """
        Calculates weights in linear gradients.
        """
        tensor = super(MultilinearDiffusionMask, self).calculate(feather_ratio)
        latent_length = int(self.ratio * self.width)
        for i in range(latent_length):
            feathered = torch.tensor(i / latent_length)
            if not self.unfeather_left:
                if self.frames is None:
                    tensor[:, :, :, i] = torch.minimum(tensor[:, :, :, i], feathered)
                else:
                    tensor[:, :, :, :, i] = torch.minimum(tensor[:, :, :, :, i], feathered)
            if not self.unfeather_top:
                if self.frames is None:
                    tensor[:, :, i, :] = torch.minimum(tensor[:, :, i, :], feathered)
                else:
                    tensor[:, :, :, i, :] = torch.minimum(tensor[:, :, :, i, :], feathered)
            if not self.unfeather_right:
                if self.frames is None:
                    tensor[:, :, :, self.width - i - 1] = torch.minimum(
                        tensor[:, :, :, self.width - i - 1],
                        feathered
                    )
                else:
                    tensor[:, :, :, :, self.width - i - 1] = torch.minimum(
                        tensor[:, :, :, :, self.width - i - 1],
                        feathered
                    )
            if not self.unfeather_bottom:
                if self.frames is None:
                    tensor[:, :, self.height - i - 1, :] = torch.minimum(
                        tensor[:, :, self.height - i - 1, :],
                        feathered
                    )
                else:
                    tensor[:, :, :, self.height - i - 1, :] = torch.minimum(
                        tensor[:, :, :, self.height - i - 1, :],
                        feathered
                    )
        if self.frames is not None:
            frame_length = int(self.ratio * self.frames)
            for i in range(frame_length):
                feathered = torch.tensor(i / frame_length)
                if not self.unfeather_start:
                    tensor[:, :, i, :, :] = torch.minimum(tensor[:, :, i, :, :], feathered)
                if not self.unfeather_end:
                    tensor[:, :, self.frames - i - 1, :, :] = torch.minimum(
                        tensor[:, :, self.frames - i - 1, :, :], 
                        feathered
                    )

        return tensor

@dataclass(frozen=True)
class GaussianDiffusionMask(DiffusionMask):
    """
    Feathers the edges and corners using gaussian probability.
    """
    deviation: float = 0.01

    def calculate(self, feather_ratio: float = 0.125) -> torch.Tensor:
        """
        Calculates weights with a gaussian distribution
        """
        midpoint = (self.width - 1) / 2
        x_probabilities = [
            exp(-(x - midpoint) * (x - midpoint) / (self.width * self.width) / (2 * self.deviation)) / sqrt(2 * pi * self.deviation)
            for x in range(self.width)
        ]
        midpoint = (self.height - 1) / 2
        y_probabilities = [
            exp(-(y - midpoint) * (y - midpoint) / (self.height * self.height) / (2 * self.deviation)) / sqrt(2 * pi * self.deviation)
            for y in range(self.height)
        ]

        if self.frames is None:
            weights = np.outer(y_probabilities, x_probabilities)
            weights = torch.tile(torch.tensor(weights), (self.batch, self.dim, 1, 1)) # type: ignore
        else:
            midpoint = (self.frames - 1) / 2
            z_probabilities = [
                exp(-(z - midpoint) * (z - midpoint) / (self.frames * self.frames) / (2 * self.deviation)) / sqrt(2 * pi * self.deviation)
                for z in range(self.frames)
            ]
            weights = np.einsum('i,j,k', z_probabilities, y_probabilities, x_probabilities)
            weights = torch.tile(torch.tensor(weights), (self.batch, self.dim, 1, 1, 1)) # type: ignore

        tmp = np.outer(y_probabilities, x_probabilities)
        tmp = torch.tile(torch.tensor(tmp), (self.batch, self.dim, 1, 1)) # type: ignore
        return self.unfeather(weights, feather_ratio) # type: ignore

@dataclass
class MaskWeightBuilder:
    """
    A class for computing blending masks given dimensions and some optional parameters

    Stores masks on the device for speed. Be sure to free memory when no longer needed.
    """
    device: Union[str, torch.device]
    dtype: torch.dtype

    constant_weights: Dict[DiffusionMask, torch.Tensor] = field(default_factory=dict)
    multilinear_weights: Dict[MultilinearDiffusionMask, torch.Tensor] = field(default_factory=dict)
    gaussian_weights: Dict[GaussianDiffusionMask, torch.Tensor] = field(default_factory=dict)
    unfeather_ratio: float = 1 / 8

    def clear(self) -> None:
        """
        Clears all stored tensors
        """
        for key in list(self.constant_weights.keys()):
            del self.constant_weights[key]
        for key in list(self.multilinear_weights.keys()):
            del self.multilinear_weights[key]
        for key in list(self.gaussian_weights.keys()):
            del self.gaussian_weights[key]

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
        batch: int,
        dim: int,
        width: int,
        height: int,
        frames: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculates the constant mask. No feathering.
        """
        mask = DiffusionMask(
            batch=batch,
            dim=dim,
            frames=frames,
            width=width,
            height=height
        )
        if mask not in self.constant_weights:
            self.constant_weights[mask] = mask.calculate(self.unfeather_ratio).to(
                dtype=self.dtype,
                device=self.device
            )
        return self.constant_weights[mask]

    def multilinear(
        self,
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
        ratio: float = 0.125,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculates the multilinear mask.
        """
        mask = MultilinearDiffusionMask(
            batch=batch,
            dim=dim,
            frames=frames,
            width=width,
            height=height,
            unfeather_left=unfeather_left,
            unfeather_top=unfeather_top,
            unfeather_right=unfeather_right,
            unfeather_bottom=unfeather_bottom,
            unfeather_start=unfeather_start,
            unfeather_end=unfeather_end,
            ratio=ratio
        )
        if mask not in self.multilinear_weights:
            self.multilinear_weights[mask] = mask.calculate(self.unfeather_ratio).to(
                dtype=self.dtype,
                device=self.device
            )
        return self.multilinear_weights[mask]

    def gaussian(
        self,
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
        deviation: float = 0.01,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Calculates the gaussian mask, optionally unfeathered.
        """
        mask = GaussianDiffusionMask(
            batch=batch,
            dim=dim,
            frames=frames,
            width=width,
            height=height,
            unfeather_left=unfeather_left,
            unfeather_top=unfeather_top,
            unfeather_right=unfeather_right,
            unfeather_bottom=unfeather_bottom,
            unfeather_start=unfeather_start,
            unfeather_end=unfeather_end,
            deviation=deviation
        )
        if mask not in self.gaussian_weights:
            self.gaussian_weights[mask] = mask.calculate(self.unfeather_ratio).to(
                dtype=self.dtype,
                device=self.device
            )
        return self.gaussian_weights[mask]

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
    ) -> torch.Tensor:
        """
        Calculates a mask depending on the method requested.
        """
        if mask_type == "constant":
            get_mask = self.constant
        elif mask_type == "multilinear":
            get_mask = self.multilinear
        elif mask_type == "gaussian":
            get_mask = self.gaussian
        else:
            raise AttributeError(f"Unknown mask type {mask_type}")

        return get_mask(
            batch=batch,
            dim=dim,
            width=width,
            height=height,
            frames=frames,
            unfeather_left=unfeather_left,
            unfeather_top=unfeather_top,
            unfeather_right=unfeather_right,
            unfeather_bottom=unfeather_bottom,
            unfeather_start=unfeather_start,
            unfeather_end=unfeather_end,
            **kwargs
        )

def interpolate_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    spherical: bool = True,
    threshold: float = 0.9995
) -> torch.Tensor:
    """
    Interpolates tensors in the desired method
    """
    if spherical:
        return interpolate_tensors_spherical(a, b, alpha, threshold)
    return interpolate_tensors_linear(a, b, alpha)

def interpolate_tensors_linear(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Performs a linear tensor interpolation
    """
    return (1.0 - alpha) * a + alpha * b

def interpolate_tensors_spherical(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    threshold: float = 0.9995
) -> torch.Tensor:
    """
    Performs a spherical linear tensor interpolation (slerp)
    """
    dot = ((a / a.norm()) * (b / b.norm())).sum()
    if dot > threshold:
        return interpolate_tensors_linear(a, b, alpha)
    omega = dot.acos()
    return (((1.0 - alpha) * omega.sin() * a + (alpha * omega).sin() * b) / omega.sin())
