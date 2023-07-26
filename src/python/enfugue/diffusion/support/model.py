from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class SupportModel:
    """
    Provides a base class for AI models that support diffusion.
    """

    def __init__(self, model_dir: str, device: torch.device, dtype: torch.dtype) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype

    @contextmanager
    def context(self) -> Iterator[None]:
        """
        Cleans torch memory after processing.
        """
        yield
        if self.device.type == "cuda":
            import torch
            import torch.cuda

            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            import torch
            import torch.mps

            torch.mps.empty_cache()
        gc.collect()
