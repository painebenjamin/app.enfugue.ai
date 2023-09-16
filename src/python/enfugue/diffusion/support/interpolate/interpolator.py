from __future__ import annotations
import os
import numpy as np

from typing import Any, Iterator, TYPE_CHECKING

from PIL import Image
from contextlib import contextmanager

if TYPE_CHECKING:
    from enfugue.diffusion.support.interpolate.model import InterpolatorModel # type: ignore[attr-defined]

from enfugue.diffusion.support.model import SupportModel
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

__all__ = ["PoseDetector"]

class InterpolatorImageProcessor:
    """
    Holds a reference to the interpolator and a callable
    """
    def __init__(self, interpolator: InterpolatorModel, **kwargs: Any) -> None:
        self.interpolator = interpolator

    def __call__(
        self,
        left: Image.Image,
        right: Image.Image,
        alpha: float
    ) -> Image:
        """
        Processes two images and returns the interpolated image.
        """
        left_data = np.asarray(left.convert("RGB")).astype(np.float32) / _UINT8_MAX_F
        right_data = np.asarray(right.convert("RGB")).astype(np.float32) / _UINT8_MAX_F
        mid_data = self.interpolator(
            np.expand_dims(left_data, axis=0),
            np.expand_dims(right_data, axis=0),
            np.full(shape=(1,), fill_value=alpha, dtype=np.float32)
        )[0]
        mid_data = (
            np.clip(mid_data * _UINT8_MAX_F, 0.0, _UINT8_MAX_F) + 0.5
        ).astype(np.uint8)
        return Image.fromarray(mid_data)

class Interpolator(SupportModel):
    """
    Uses OpenPose to predict human poses.
    """
    
    STYLE_MODEL_REPO = "akhaliq/frame-interpolation-film-style"

    @property
    def style_model_path(self) -> str:
        """
        Gets the style model path
        """
        if not hasattr(self, "_style_model_path"):
            from huggingface_hub import snapshot_download
            self._style_model_path = os.path.join(
                self.model_dir,
                "models--" + self.STYLE_MODEL_REPO.replace("/", "--")
            )
            if not os.path.exists(self._style_model_path):
                os.makedirs(self._style_model_path)
            if not os.path.exists(os.path.join(self._style_model_path, "saved_model.pb")):
                snapshot_download(
                    self.STYLE_MODEL_REPO,
                    local_dir=self._style_model_path,
                    local_dir_use_symlinks=False
                )
        return self._style_model_path

    @contextmanager
    def interpolate(self) -> Iterator[InterpolatorImageProcessor]:
        """
        Gets and runs the interpolator.
        """
        from enfugue.diffusion.support.interpolate.model import InterpolatorModel # type: ignore[attr-defined]

        with self.context():
            model = InterpolatorModel(self.style_model_path)
            processor = InterpolatorImageProcessor(model)
            yield processor
            del model
