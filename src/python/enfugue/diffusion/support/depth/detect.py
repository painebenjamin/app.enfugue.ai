from __future__ import annotations

import cv2
import numpy as np

from typing import Iterator, Callable, Any, TYPE_CHECKING

from contextlib import contextmanager
from PIL import Image

from enfugue.diffusion.util import ComputerVision
from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

if TYPE_CHECKING:
    import torch

__all__ = ["DepthDetector"]

class MidasImageProcessor(SupportModelImageProcessor):
    """
    Stores the depth model and transform function
    """
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        transform: Callable[[np.ndarray], torch.Tensor],
        **kwargs: Any
    ) -> None:
        super(MidasImageProcessor, self).__init__(**kwargs)
        self.device = device
        self.model = model
        self.transform = transform

    def depth(self, image: Image.Image) -> torch.Tensor:
        """
        Runs the depth prediction
        """
        import torch
        with torch.no_grad():
            image = ComputerVision.convert_image(image)
            batch = self.transform(image).to(self.device)
            return self.model(batch)

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth prediction then returns to an image
        """
        import torch
        output = self.depth(image)
        output = torch.nn.functional.interpolate(
            output.unsqueeze(1),
            size=tuple(reversed(image.size)),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = output.cpu().numpy()
        output = (output * 255 / np.max(output)).astype(np.uint8)
        return Image.fromarray(output)

class NormalImageProcessor(MidasImageProcessor):
    """
    Extends the depth processor to perform normal estimation
    """
    @staticmethod
    def from_midas(midas_processor: MidasImageProcessor) -> NormalImageProcessor:
        """
        Create this from a midas processor
        """
        return NormalImageProcessor(
            device=midas_processor.device,
            model=midas_processor.model,
            transform=midas_processor.transform
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth prediction then transforms into a normal
        """
        size = image.size
        image = self.depth(image).cpu()[0].numpy()
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
        return Image.fromarray(image).resize(size)

class DepthDetector(SupportModel):
    """
    Uses MiDaS v2 to predict depth.
    Uses depth prediction to generate normal maps.
    """

    MIDAS_MODEL_TYPE = "DPT_Hybrid"
    MIDAS_TRANSFORM_TYPE = "transforms"
    MIDAS_PATH = "intel-isl/MiDaS"

    @contextmanager
    def midas(self) -> Iterator[MidasImageProcessor]:
        """
        Executes MiDaS depth estimation
        """
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

            processor = MidasImageProcessor(
                model=model,
                transform=transform,
                device=self.device
            )
            yield processor
            del model
            del transforms
            del processor

    @contextmanager
    def normal(self) -> Iterator[NormalImageProcessor]:
        """
        Executes normal estimation via midas depth detection
        """
        with self.midas() as midas:
            processor = NormalImageProcessor.from_midas(midas)
            yield processor
            del processor
