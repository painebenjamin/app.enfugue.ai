from __future__ import annotations

from typing import Dict, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "load_ckpt_state_dict",
    "load_safetensor_state_dict",
    "load_state_dict",
]

def load_ckpt_state_dict(path: str) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
    """
    Loads a state dictionary from a .ckpt (old-style) file
    """
    import torch
    return torch.load(path, map_location="cpu")

def load_safetensor_state_dict(path: str) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
    """
    Loads a state dictionary from a .safetensor(s) (new-style) file
    """
    from safetensors import safe_open

    checkpoint = {}
    with safe_open(path, framework="pt", device="cpu") as f: # type: ignore[attr-defined]
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
    return checkpoint

def load_state_dict(path: str) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
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
