from __future__ import annotations

import cv2
import PIL
import numpy as np

from enfugue.util import check_download_to_dir
from enfugue.diffusion.support.model import SupportModel
from enfugue.diffusion.support.vision import ComputerVision

__all__ = [
    "EdgeDetector"
]

class EdgeDetector(SupportModel):
    """
    Provides edge detection methods
    """

    HED_PROTOTXT = "https://github.com/ashukid/hed-edge-detector/raw/master/deploy.prototxt"
    HED_CAFFEMODEL = "https://github.com/ashukid/hed-edge-detector/raw/master/hed_pretrained_bsds.caffemodel"
    HED_MEAN = (104.00698793, 116.66876762, 122.67891434)
    PRETRAINED_PATH = "lllyasviel/Annotators"

    @property
    def hed_prototxt(self) -> str:
        """
        Gets the local path to the HED prototxt.
        """
        return check_download_to_dir(self.HED_PROTOTXT, self.model_dir)

    @property
    def hed_caffemodel(self) -> str:
        """
        Gets the local path to the HED caffemodel.
        """
        return check_download_to_dir(self.HED_CAFFEMODEL, self.model_dir)
    
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
