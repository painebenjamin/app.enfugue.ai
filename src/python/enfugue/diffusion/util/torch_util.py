from typing import Dict, Type, Union

import torch
import numpy as np

__all__ = [
    "cuda_available",
    "tensorrt_available",
    "mps_available",
    "directml_available",
    "get_optimal_device",
    "DTypeConverter",
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
