from __future__ import annotations

import os
import numpy as np
import requests

from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

from typing import Iterator, Any, TYPE_CHECKING

from contextlib import contextmanager

from enfugue.util import logger

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from enfugue.diffusion.support.background.u2net.helper import Net # type: ignore

__all__ = ["BackgroundRemover"]

class BackgroundRemoverImageProcessor(SupportModelImageProcessor):
    """
    Used to detect line art
    """
    def __init__(self, model: Net, device: torch.device, **kwargs: Any) -> None:
        super(BackgroundRemoverImageProcessor, self).__init__(**kwargs)
        self.model = model
        self.device = device

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Runs the detector.
        """
        from enfugue.diffusion.support.background.u2net.detect import predict # type: ignore
        from PIL import Image
        image = image.convert("RGB") # Remove alpha channel
        mask = predict(self.model, np.array(image), self.device).convert("L")
        empty = Image.new("RGBA", (image.size), 0)
        return Image.composite(image, empty, mask.resize(image.size, Image.LANCZOS))

class BackgroundRemover(SupportModel):
    """
    Used to remove backgrounds from images automatically
    """
    U2NET_MODEL_FILE = "u2net.pth"
    U2NET_MODEL_PATHS = [
        "https://github.com/nadermx/backgroundremover/raw/main/models/u2aa",
        "https://github.com/nadermx/backgroundremover/raw/main/models/u2ab",
        "https://github.com/nadermx/backgroundremover/raw/main/models/u2ac",
        "https://github.com/nadermx/backgroundremover/raw/main/models/u2ad",
    ]

    @property
    def u2net_model_path(self) -> str:
        """
        Gets the U2Net model path.
        This comes in multiple files, so we don't use the provided method to download.
        """
        try:
            return self.get_model_file(self.U2NET_MODEL_FILE)
        except IOError:
            target_model = os.path.join(self.model_dir, self.U2NET_MODEL_FILE)
            with open(target_model, "wb") as output:
                for i, path in enumerate(self.U2NET_MODEL_PATHS):
                    logger.debug(f"Downloading part {i+1} of {target_model}")
                    output.write(requests.get(path).content)
            return target_model

    @contextmanager
    def remover(self) -> Iterator[BackgroundRemoverImageProcessor]:
        """
        Runs the background remover on an image.
        """
        from enfugue.diffusion.support.background.u2net.helper import Net # type: ignore
        with self.context():
            net = Net("u2net", self.u2net_model_path)
            net.to(self.device)
            processor = BackgroundRemoverImageProcessor(net, self.device)
            yield processor
            del processor
            del net
