from __future__ import annotations

import os
from enfugue.diffusion.support.model import SupportModel

from typing import Iterator, Any, TYPE_CHECKING

from contextlib import contextmanager

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from enfugue.diffusion.support.face.insight.app.face_analysis import FaceAnalysis # type: ignore

__all__ = ["FaceAnalyzer"]

class FaceAnalyzerImageProcessor:
    """
    Used to detect line art
    """
    def __init__(self, analyzer: FaceAnalysis, **kwargs: Any) -> None:
        super(FaceAnalyzerImageProcessor, self).__init__(**kwargs)
        self.analyzer = analyzer

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Runs the detector.
        """
        import torch
        from enfugue.diffusion.util import ComputerVision
        image = ComputerVision.convert_image(image)
        faces = self.analyzer.get(image)
        return torch.cat([
            torch.from_numpy(face.normed_embedding).unsqueeze(0)
            for face in faces
        ], dim=0)

class FaceAnalyzer(SupportModel):
    """
    Uses to analyze facial features.
    """
    BUFFALO_MODEL_PATH = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

    @property
    def insightface_model_dir(self) -> str:
        """
        Gets the insightface model directory, downloading it if needed.
        """
        model_dir = os.path.join(self.model_dir, "buffalo_l")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            model_zip_file = self.get_model_file(self.BUFFALO_MODEL_PATH)
            from zipfile import ZipFile
            with ZipFile(model_zip_file) as fh:
                fh.extractall(model_dir)
            os.remove(model_zip_file)
        return model_dir

    @contextmanager
    def insightface(self) -> Iterator[FaceAnalyzerImageProcessor]:
        """
        Runs the line art detector on an image.
        """
        from enfugue.diffusion.support.face.insight.app.face_analysis import FaceAnalysis # type: ignore
        with self.context():
            analyzer = FaceAnalysis(self.insightface_model_dir)
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            processor = FaceAnalyzerImageProcessor(analyzer)
            yield processor
            del processor
            del analyzer
