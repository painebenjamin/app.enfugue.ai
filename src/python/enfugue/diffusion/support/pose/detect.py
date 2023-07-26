from __future__ import annotations

import PIL

from enfugue.diffusion.support.model import SupportModel

__all__ = ["PoseDetector"]


class PoseDetector(SupportModel):
    """
    Uses OpenPose to predict human poses.
    """

    def detect(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Gets and runs the detector.
        """
        from enfugue.diffusion.support.pose.helper import OpenposeDetector  # type: ignore

        with self.context():
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=self.model_dir)
            detector.to(self.device)
            result = detector(image, hand_and_face=True)
            del detector
            return result
