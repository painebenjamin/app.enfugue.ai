from __future__ import annotations

import cv2
import PIL
import numpy as np

from enfugue.diffusion.support.model import SupportModel

__all__ = ["EdgeDetector"]


class EdgeDetector(SupportModel):
    """
    Provides edge detection methods
    """

    PRETRAINED_PATH = "lllyasviel/Annotators"

    @staticmethod
    def canny(image: PIL.Image.Image, lower: int = 100, upper: int = 200) -> PIL.Image.Image:
        """
        Runs canny edge detection on an image. This one isn't AI.
        """
        canny = cv2.Canny(np.array(image), lower, upper)[:, :, None]
        return PIL.Image.fromarray(np.concatenate([canny, canny, canny], axis=2))

    def pidi(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Runs soft-edge detection using PIDI
        """
        from enfugue.diffusion.support.edge.pidi import PidiNetDetector  # type: ignore

        with self.context():
            detector = PidiNetDetector.from_pretrained(self.PRETRAINED_PATH, cache_dir=self.model_dir)
            detector.to(self.device)
            result = detector(image, safe=True)
            del detector
            return result

    def hed(self, image: PIL.Image.Image, scribble: bool = False) -> PIL.Image.Image:
        """
        Runs holistically-nested edge detection on an image.
        """
        from enfugue.diffusion.support.edge.hed import HEDDetector  # type: ignore

        with self.context():
            detector = HEDDetector.from_pretrained(self.PRETRAINED_PATH)
            detector.to(self.device)
            result = detector(image, scribble=scribble)
            del detector
            return result
