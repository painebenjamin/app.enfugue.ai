from __future__ import annotations

import gc
import PIL

from typing import Iterator, TYPE_CHECKING

from contextlib import contextmanager

from enfugue.diffusion.pose.helper import OpenposeDetector  # type: ignore[attr-defined]

if TYPE_CHECKING:
    import torch


class PoseDetector:
    """
    Uses OpenPose to predict human poses.
    """

    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.model_dir = model_dir
        self.device = device

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

    def detect(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Gets and runs the detector.
        """
        with self.context():
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=self.model_dir)
            detector.to(self.device)
            result = detector(image, hand_and_face=True)
            del detector
            return result
