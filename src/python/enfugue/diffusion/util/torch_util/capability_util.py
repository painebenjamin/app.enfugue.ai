from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device as Device

__all__ = [
    "cuda_available",
    "tensorrt_available",
    "mps_available",
    "directml_available",
    "get_optimal_device",
    "get_ram_info",
    "get_vram_info",
    "empty_cache",
]

def tensorrt_available() -> bool:
    """
    Returns true if TensorRT is available.
    """
    try:
        import tensorrt
        tensorrt  # silence importchecker
        import onnx
        onnx # silence importchecker
        import onnx_graphsurgeon
        onnx_graphsurgeon # silence importchecker
        return True
    except:
        return False

def cuda_available() -> bool:
    """
    Returns true if CUDA is available.
    """
    import torch
    return torch.cuda.is_available() and torch.backends.cuda.is_built()

def mps_available() -> bool:
    """
    Returns true if MPS is available.
    """
    import torch
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

def get_optimal_device() -> Device:
    """
    Gets the optimal device based on availability.
    """
    import torch
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
    import torch
    return torch.cuda.mem_get_info()

def empty_cache() -> None:
    """
    Empties caches to clear memory.
    """
    if cuda_available():
        import torch
        import torch.cuda
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif mps_available():
        import torch
        import torch.mps
        torch.mps.empty_cache()
        torch.mps.synchronize()
    import gc
    gc.collect()
