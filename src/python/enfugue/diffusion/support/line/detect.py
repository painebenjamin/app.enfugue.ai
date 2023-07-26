from __future__ import annotations

import cv2
import PIL
import numpy as np

from enfugue.util import check_download_to_dir
from enfugue.diffusion.support.model import SupportModel
from enfugue.diffusion.support.vision import ComputerVision

__all__ = ["LineDetector"]


class LineDetector(SupportModel):
    """
    Uses to predict human poses.
    """

    MLSD_WEIGHTS = "https://github.com/lhwcv/mlsd_pytorch/raw/main/models/mlsd_large_512_fp32.pth"
    LINEART_PATH = "lllyasviel/Annotators"

    @property
    def mlsd_weights(self) -> str:
        """
        Gets the local path to the MLSD weights file.
        """
        return check_download_to_dir(self.MLSD_WEIGHTS, self.model_dir)

    def detect(self, image: PIL.Image.Image, anime: bool = False) -> PIL.Image.Image:
        """
        Runs the line art detector on an image.
        """
        if anime:
            from enfugue.diffusion.support.line.anime import LineartAnimeDetector  # type: ignore

            detector_class = LineartAnimeDetector
        else:
            from enfugue.diffusion.support.line.art import LineartDetector  # type: ignore

            detector_class = LineartDetector
        with self.context():
            detector = detector_class.from_pretrained(self.LINEART_PATH, cache_dir=self.model_dir)
            detector.to(self.device)
            result = detector(image)
            del detector
            return result

    def mlsd(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Runs Mobile Line Segment Detection (MLSD) on an image.
        """
        import torch
        from enfugue.diffusion.support.line.mlsd import MLSD, pred_lines  # type: ignore

        with self.context():
            model = MLSD().to(self.device).eval()
            model.load_state_dict(torch.load(self.mlsd_weights, map_location=self.device), strict=True)
            cv2_image = ComputerVision.convert_image(image)
            cv2_image = cv2.resize(cv2_image, (512, 512))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            lines = pred_lines(cv2_image, model, [512, 512], 0.1, 20, self.device)
            cv2_image = np.zeros((512, 512, 3), np.uint8)

            for line in lines:
                cv2.line(
                    cv2_image,
                    (int(line[0]), int(line[1])),
                    (int(line[2]), int(line[3])),
                    (255, 255, 255),
                    1,
                    16,
                )
            del model
            return ComputerVision.revert_image(cv2_image).resize(image.size)
