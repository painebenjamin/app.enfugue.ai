# adapted from https://github.com/TianxingWu/FreeInit/blob/master/freeinit_utils.py
from __future__ import annotations
from typing import Union, Literal, Tuple, List, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from torch import Tensor, Size, device as Device

__all__ = [
    "freq_mix_3d",
    "get_freq_filter",
    "gaussian_low_pass_filter",
    "ideal_low_pass_filter",
    "box_low_pass_filter",
    "butterworth_low_pass_filter"
]

def freq_mix_3d(
    sample: Tensor,
    noise: Tensor,
    low_pass_filter: Tensor
) -> Tensor:
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        low_pass_filter: low pass filter
    """
    import torch.fft as fft
    # FFT
    x_freq = fft.fftn(sample, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    high_pass_filter = 1 - low_pass_filter
    x_freq_low = x_freq * low_pass_filter
    noise_freq_high = noise_freq * high_pass_filter
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed

def get_freq_filter(
    shape: Union[List[int], Tuple[int], Size],
    device: Union[str, Device],
    filter_type: Literal["gaussian", "ideal", "box", "butterworth"],
    n: int=4,
    d_s: float=0.25,
    d_t: float=0.25
) -> Tensor:
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(
    shape: Union[List[int], Tuple[int], Size],
    d_s: float=0.25,
    d_t: float=0.25
) -> Tensor:
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    import torch
    T, H, W = shape[-3], shape[-2], shape[-1] # type: ignore[misc]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask

def butterworth_low_pass_filter(
    shape: Union[List[int], Tuple[int], Size],
    n: int=4,
    d_s: float=0.25,
    d_t: float=0.25
) -> Tensor:
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    import torch
    T, H, W = shape[-3], shape[-2], shape[-1] # type: ignore[misc]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask

def ideal_low_pass_filter(
    shape: Union[List[int], Tuple[int], Size],
    d_s: float=0.25,
    d_t: float=0.25
) -> Tensor:
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    import torch
    T, H, W = shape[-3], shape[-2], shape[-1] # type: ignore[misc]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask

def box_low_pass_filter(
    shape: Union[List[int], Tuple[int], Size],
    d_s: float=0.25,
    d_t: float=0.25
) -> Tensor:
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    import torch
    T, H, W = shape[-3], shape[-2], shape[-1] # type: ignore[misc]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask
