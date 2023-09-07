from __future__ import annotations

from typing import Any, Callable, Iterator, TYPE_CHECKING

from contextlib import contextmanager

if TYPE_CHECKING:
    from PIL.Image import Image

from enfugue.diffusion.support.model import SupportModel, SupportModelImageProcessor

__all__ = ["PoseDetector"]

class OpenPoseImageProcessor(SupportModelImageProcessor):
    """
    Uses OpenPose to detect human poses, hands, and faces
    """
    def __init__(self, detector: Callable, **kwargs: Any) -> None:
        super(OpenPoseImageProcessor, self).__init__(**kwargs)
        self.detector = detector

    def __call__(self, image: Image) -> Image:
        """
        Calls the detector
        """
        return self.detector(
            image,
            include_body=True,
            include_hand=True,
            include_face=True
        ).resize(image.size)

class DWPoseImageProcessor(SupportModelImageProcessor):
    """
    Uses OpenPose to detect human poses, hands, and faces
    """
    def __init__(self, detector: Callable[[Image], Image], **kwargs: Any) -> None:
        super(DWPoseImageProcessor, self).__init__(**kwargs)
        self.detector = detector

    def __call__(self, image: Image) -> Image:
        """
        Calls the detector
        """
        return self.detector(image).resize(image.size)

class PoseDetector(SupportModel):
    """
    Uses OpenPose to predict human poses.
    """

    @contextmanager
    def dwpose(self) -> Iterator[SupportModelImageProcessor]:
        """
        Gets and runs the DWPose detector.
        """
        from enfugue.diffusion.support.pose.dwpose import DWposeDetector  # type: ignore

        with self.context():
            detector = DWposeDetector()
            detector.to(self.device)
            processor = DWPoseImageProcessor(detector)
            yield processor
            del processor
            del detector

    @contextmanager
    def openpose(self) -> Iterator[SupportModelImageProcessor]:
        """
        Gets and runs the OpenPose detector.
        """
        from enfugue.diffusion.support.pose.openpose.helper import OpenposeDetector  # type: ignore

        with self.context():
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=self.model_dir)
            detector.to(self.device)
            processor = OpenPoseImageProcessor(detector)
            yield processor
            del processor
            del detector

    @contextmanager
    def best(self) -> Iterator[SupportModelImageProcessor]:
        """
        Uses DWPose if available, otherwise uses OpenPose.
        """
        try:
            from enfugue.diffusion.support.pose.dwpose import DWposeDetector  # type: ignore
            context = self.dwpose
        except ImportError:
            context = self.openpose
        with context() as processor:
            yield processor
