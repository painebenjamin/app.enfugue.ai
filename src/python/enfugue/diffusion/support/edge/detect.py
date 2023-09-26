from __future__ import annotations

import cv2
import numpy as np

from typing import Iterator, Callable, Any

from contextlib import contextmanager
from PIL import Image

from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

__all__ = ["EdgeDetector"]


class CannyImageProcessor(SupportModelImageProcessor):
    """
    Simply stores thresholds for processor
    """
    def __init__(self, lower: int = 100, upper: int = 200, **kwargs: Any) -> None:
        super(CannyImageProcessor, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Call the method (no model)
        """
        canny = cv2.Canny(np.array(image), self.lower, self.upper)[:, :, None]
        return Image.fromarray(np.concatenate([canny, canny, canny], axis=2))

class PidiImageProcessor(SupportModelImageProcessor):
    """
    Stores a reference to the Pidi detector.
    """
    def __init__(self, detector: Callable, **kwargs: Any) -> None:
        super(PidiImageProcessor, self).__init__(**kwargs)
        self.detector = detector

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Call the model
        """
        resolution = min(image.size)
        return self.detector(
            image,
            image_resolution=resolution,
            detect_resolution=resolution,
            safe=True
        ).resize(image.size)

class HEDImageProcessor(SupportModelImageProcessor):
    """
    Stores a reference to the HED detector.
    """
    def __init__(self, detector: Callable, scribble: bool = False, **kwargs: Any) -> None:
        super(HEDImageProcessor, self).__init__(**kwargs)
        self.detector = detector
        self.scribble = scribble

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Call the model
        """
        resolution = min(image.size)
        return self.detector(
            image,
            detect_resolution=resolution,
            image_resolution=resolution,
            scribble=self.scribble
        ).resize(image.size)

class EdgeDetector(SupportModel):
    """
    Provides edge detection methods
    """

    PIDINET_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
    HEDNET_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"

    @contextmanager
    def canny(self, lower: int = 100, upper: int = 200) -> Iterator[SupportModelImageProcessor]:
        """
        Runs canny edge detection on an image. This one isn't AI.
        """
        with self.context():
            processor = CannyImageProcessor(lower=lower, upper=upper)
            yield processor
            del processor

    @contextmanager
    def pidi(self) -> Iterator[PidiImageProcessor]:
        """
        Runs soft-edge detection using Pidi
        """
        from enfugue.diffusion.support.edge.pidi import PidiNetDetector  # type: ignore

        with self.context():
            pidinet_path = self.get_model_file(self.PIDINET_PATH)
            detector = PidiNetDetector.from_pretrained(pidinet_path)
            detector.to(self.device)
            processor = PidiImageProcessor(detector)
            yield processor
            del processor
            del detector

    @contextmanager
    def hed(self) -> Iterator[HEDImageProcessor]:
        """
        Runs holistically-nested edge detection on an image.
        """
        from enfugue.diffusion.support.edge.hed import HEDDetector  # type: ignore

        with self.context():
            hednet_path = self.get_model_file(self.HEDNET_PATH)
            detector = HEDDetector.from_pretrained(hednet_path)
            detector.to(self.device)
            processor = HEDImageProcessor(detector)
            yield processor
            del processor
            del detector
    
    @contextmanager
    def scribble(self) -> Iterator[HEDImageProcessor]:
        """
        Runs holistically-nested edge detection on an image, modified to resemble scribbles.
        """
        from enfugue.diffusion.support.edge.hed import HEDDetector  # type: ignore

        with self.context():
            hednet_path = self.get_model_file(self.HEDNET_PATH)
            detector = HEDDetector.from_pretrained(hednet_path)
            detector.to(self.device)
            processor = HEDImageProcessor(detector, scribble=True)
            yield processor
            del processor
            del detector
