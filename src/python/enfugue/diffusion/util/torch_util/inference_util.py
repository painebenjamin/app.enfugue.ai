from __future__ import annotations

from typing import Any, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["apply_freeu"]

def apply_freeu(
    resolution_idx: Optional[int],
    hidden_states: Tensor,
    res_hidden_states: Tensor,
    s1: float = 1.0,
    s2: float = 1.0,
    b1: float = 1.0,
    b2: float = 1.0,
    use_structure_scaling: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    This is the diffusers implementation of apply_freeu but optionally
    with the structure-based scaling proposed in version 2 of the paper
    """
    import torch
    from diffusers.utils.torch_utils import fourier_filter
    b, s = None, None

    if resolution_idx == 0:
        # First layer up blocks
        b = b1
        s = s1
    elif resolution_idx == 1:
        # Second layer up blocks
        b = b2
        s = s2
    
    if b is not None and s is not None:
        num_half_channels = hidden_states.shape[1] // 2

        if use_structure_scaling:
            hidden_mean = hidden_states.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * ((b - 1) * hidden_mean + 1)
        else:
            hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * b

        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=s)

    return hidden_states, res_hidden_states
