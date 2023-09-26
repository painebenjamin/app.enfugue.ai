from __future__ import annotations

import cv2
import numpy as np

from enfugue.diffusion.util import ComputerVision
from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

from typing import Iterator, Any, Callable, TYPE_CHECKING

from contextlib import contextmanager

if TYPE_CHECKING:
    import torch
    from PIL import Image

__all__ = ["LineDetector"]

class LineArtImageProcessor(SupportModelImageProcessor):
    """
    Used to detect line art
    """
    def __init__(self, detector: Callable[[Image.Image], Image], **kwargs: Any) -> None:
        super(LineArtImageProcessor, self).__init__(**kwargs)
        self.detector = detector

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Runs the detector.
        """
        resolution = min(image.size)
        return self.detector( # type: ignore [call-arg]
            image,
            detect_resolution=resolution,
            image_resolution=resolution
        ).resize(image.size)

class MLSDImageProcessor(SupportModelImageProcessor):
    """
    Used to detect straight lines
    """
    def __init__(self, device: torch.device, model: torch.nn.Module, **kwargs: Any) -> None:
        super(MLSDImageProcessor, self).__init__(**kwargs)
        self.device = device
        self.model = model

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Runs the detector.
        """
        from enfugue.diffusion.support.line.mlsd import pred_lines  # type: ignore
        width, height = image.size
        
        image = ComputerVision.convert_image(image)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lines = pred_lines(image, self.model, [512, 512], 0.1, 20, self.device)

        image = np.zeros((height, width, 3), np.uint8)
        w_ratio = width / 512
        h_ratio = height / 512

        for line in lines:
            cv2.line(
                image,
                (int(line[0] * w_ratio), int(line[1] * h_ratio)),
                (int(line[2] * w_ratio), int(line[3] * h_ratio)),
                (255, 255, 255),
                1,
                16,
            )

        return ComputerVision.revert_image(image).resize((width, height))

class LineDetector(SupportModel):
    """
    Uses to predict human poses.
    """

    MLSD_MODEL_PATH = "https://github.com/lhwcv/mlsd_pytorch/raw/main/models/mlsd_large_512_fp32.pth"
    LINEART_MODEL_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth"
    LINEART_COARSE_MODEL_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth"
    LINEART_ANIME_MODEL_PATH = "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth"

    @contextmanager
    def lineart(self) -> Iterator[LineArtImageProcessor]:
        """
        Runs the line art detector on an image.
        """
        from enfugue.diffusion.support.line.art import LineartDetector  # type: ignore
        with self.context():
            model_path = self.get_model_file(self.LINEART_MODEL_PATH)
            coarse_model_path = self.get_model_file(self.LINEART_COARSE_MODEL_PATH)
            detector = LineartDetector.from_pretrained(model_path, coarse_model_path)
            detector.to(self.device)
            processor = LineArtImageProcessor(detector)
            yield processor
            del processor
            del detector

    @contextmanager
    def anime(self) -> Iterator[LineArtImageProcessor]:
        """
        Runs the anime line art detector on an image.
        """
        from enfugue.diffusion.support.line.anime import LineartAnimeDetector  # type: ignore
        with self.context():
            model_path = self.get_model_file(self.LINEART_ANIME_MODEL_PATH)
            detector = LineartAnimeDetector.from_pretrained(model_path)
            detector.to(self.device)
            processor = LineArtImageProcessor(detector)
            yield processor
            del processor
            del detector

    @contextmanager
    def mlsd(self) -> Iterator[MLSDImageProcessor]:
        """
        Runs Mobile Line Segment Detection (MLSD) on an image.
        """
        import torch
        from enfugue.diffusion.support.line.mlsd import MLSD  # type: ignore

        with self.context():
            mlsd_path = self.get_model_file(self.MLSD_MODEL_PATH)
            model = MLSD().to(self.device).eval()
            model.load_state_dict(torch.load(mlsd_path, map_location=self.device), strict=True)
            processor = MLSDImageProcessor(self.device, model)
            yield processor
            del processor
            del model
