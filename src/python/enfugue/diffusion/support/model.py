from __future__ import annotations

import os
import gc

from contextlib import contextmanager

from typing import Iterator, Any, Optional, Callable, List, TYPE_CHECKING
from typing_extensions import Self

from enfugue.util import find_file_in_directory, check_download

if TYPE_CHECKING:
    from PIL.Image import Image
    import torch

class SupportModelImageProcessor:
    def __init__(self, **kwargs: Any) -> None:
        """
        Provides a base class for processing images with an AI model.
        """
        self.kwargs = kwargs

    def __enter__(self) -> Self:
        """
        Compatibility with contexts
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """
        Compatibility with contexts
        """
        return

    def __call__(self, image: Image) -> Image:
        """
        Implemented by the image processor.
        """
        raise NotImplementedError("Implementation did not override __call__")

class SupportModel:
    """
    Provides a base class for AI models that support diffusion.
    """

    process: Optional[SupportModelImageProcessor] = None
    task_callback: Optional[Callable[[str], None]] = None

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        offline: bool = False
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.offline = offline

    def get_model_file(
        self,
        uri: str,
        filename: Optional[str] = None,
        extensions: Optional[List[str]] = None,
    ) -> str:
        """
        Searches for a file in the current directory.
        If it's not found and the passed URI is HTTP, it will be downloaded.
        """
        if os.path.exists(uri):
            # File already exists right where you passed it ya silly goose
            return uri
        if filename is None:
            filename = os.path.basename(uri)
        if extensions is not None:
            basename, ext = os.path.splitext(filename)
            existing_path = find_file_in_directory(self.model_dir, basename, extensions=extensions)
        else:
            existing_path = find_file_in_directory(self.model_dir, filename)
        if existing_path is not None:
            return existing_path # Already downloaded somewhere (can be nested)
        if uri.startswith("http"):
            local_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(local_path):
                if self.offline:
                    raise IOError(f"Offline mode is enabled and could not find requested model file at {local_path}")
                elif self.task_callback is not None:
                    self.task_callback(f"Downloading {uri}")
            check_download(uri, local_path, check_size=False)
            return local_path
        raise IOError(f"Cannot retrieve model file {uri}")

    @contextmanager
    def context(self) -> Iterator[Self]:
        """
        Cleans torch memory after processing.
        """
        self.loaded = True
        yield self
        self.loaded = False
        if getattr(self, "process", None) is not None:
            del self.process
        if self.device.type == "cuda":
            import torch
            import torch.cuda

            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            import torch
            import torch.mps

            torch.mps.empty_cache()
        gc.collect()
