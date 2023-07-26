from __future__ import annotations

import cv2
import PIL
import numpy as np

from typing import Tuple

from enfugue.diffusion.support.model import SupportModel
from enfugue.diffusion.support.vision import ComputerVision

__all__ = ["DepthDetector"]


class DepthDetector(SupportModel):
    """
    Uses MiDaS v2 to predict depth.
    Uses depth prediction to generate normal maps.
    """

    MIDAS_MODEL_TYPE = "DPT_Hybrid"
    MIDAS_TRANSFORM_TYPE = "transforms"
    MIDAS_PATH = "intel-isl/MiDaS"

    def execute(self, image: PIL.Image.Image) -> Tuple[np.ndarray, PIL.Image.Image]:
        import torch

        with self.context():
            torch.hub.set_dir(self.model_dir)
            model = torch.hub.load(self.MIDAS_PATH, self.MIDAS_MODEL_TYPE)
            model.to(self.device)
            model.eval()

            transforms = torch.hub.load(self.MIDAS_PATH, self.MIDAS_TRANSFORM_TYPE)
            if "dpt" in self.MIDAS_MODEL_TYPE.lower():
                transform = transforms.dpt_transform
            else:
                transform = transforms.small_transform
            image = ComputerVision.convert_image(image)
            input_batch = transform(image).to(self.device)

            with torch.no_grad():
                depth = model(input_batch)
                pred = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = pred.cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype(np.uint8)
            image = PIL.Image.fromarray(formatted)
            return_tuple = (depth.cpu(), image)
            del model
            del transforms
            return return_tuple

    def midas(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Executes midas depth detection
        """
        return self.execute(image)[1]

    def normal(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Executes normal estimation via midas depth detection
        """
        image = self.execute(image)[0][0].numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.4

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)  # type: ignore
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)  # type: ignore
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return PIL.Image.fromarray(image)
