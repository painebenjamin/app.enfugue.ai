from __future__ import annotations

from typing import Tuple, Any, Optional, TYPE_CHECKING

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
    "debug_tensors",
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

def get_optimal_device(device_index: Optional[int] = None) -> Device:
    """
    Gets the optimal device based on availability.
    """
    import torch
    if cuda_available():
        return torch.device("cuda", 0 if device_index is None else device_index)
    elif directml_available():
        import torch_directml
        return torch_directml.device()
    elif mps_available():
        return torch.device("mps", 0 if device_index is None else device_index)
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

def debug_tensors(*args: Any, **kwargs: Any) -> None:
    """
    Logs tensors
    """
    import torch
    from enfugue.util import logger
    arg_dict = dict([
        (f"arg_{i}", arg)
        for i, arg in enumerate(args)
    ])
    for tensor_dict in [arg_dict, kwargs]:
        for key, value in tensor_dict.items():
            if isinstance(value, list) or isinstance(value, tuple):
                for i, v in enumerate(value):
                    debug_tensors(**{f"{key}_{i}": v})
            elif isinstance(value, dict):
                for k, v in value.items():
                    debug_tensors(**{f"{key}_{k}": v})
            elif isinstance(value, torch.Tensor):
                t_min, t_max = value.aminmax()
                logger.debug(f"{key} = {value.shape} ({value.dtype}) on {value.device}, min={t_min}, max={t_max}")
