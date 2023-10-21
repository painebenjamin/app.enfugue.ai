from __future__ import annotations

from typing import Iterator, Tuple, Callable, Optional, TYPE_CHECKING

from contextlib import contextmanager, ExitStack

from enfugue.diffusion.constants import CONTROLNET_LITERAL
from enfugue.diffusion.support.model import SupportModelImageProcessor

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image
    from enfugue.diffusion.support.depth import DepthDetector
    from enfugue.diffusion.support.edge import EdgeDetector
    from enfugue.diffusion.support.line import LineDetector
    from enfugue.diffusion.support.pose import PoseDetector

class PassThroughImageProcessor(SupportModelImageProcessor):
    """
    Does not process an image.
    """
    def __call__(self, image: Image) -> Image:
        return image

class ControlImageProcessor:
    """
    Amalgamates all controlnet processors.
    Allows multiple contexts at once
    """
    task_callback: Optional[Callable[[str], None]] = None
    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        offline: bool = False
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.offline = offline

    @contextmanager
    def processors(self, *controlnets: CONTROLNET_LITERAL) -> Iterator[Tuple[SupportModelImageProcessor, ...]]:
        """
        Gets any number of controlnet processors in context.
        """
        with ExitStack() as stack:
            uniques = set(controlnets)
            processors = dict([
                (
                    controlnet,
                    stack.enter_context(self.processor(controlnet))
                )
                for controlnet in uniques
            ])
            yield tuple([
                processors[controlnet]
                for controlnet in controlnets
            ])

    @contextmanager
    def processor(self, controlnet: CONTROLNET_LITERAL) -> Iterator[SupportModelImageProcessor]:
        """
        Gets one controlnet processor in context.
        """
        context: Callable
        if controlnet == "canny":
            context = self.edge_detector.canny
        elif controlnet == "pidi":
            context = self.edge_detector.pidi
        elif controlnet == "hed":
            context = self.edge_detector.hed
        elif controlnet == "scribble":
            context = self.edge_detector.scribble
        elif controlnet == "depth":
            context = self.depth_detector.midas
        elif controlnet == "normal":
            context = self.depth_detector.normal
        elif controlnet == "pose":
            context = self.pose_detector.best
        elif controlnet == "line":
            context = self.line_detector.lineart
        elif controlnet == "anime":
            context = self.line_detector.anime
        elif controlnet == "mlsd":
            context = self.line_detector.mlsd
        else:
            context = PassThroughImageProcessor
        with context() as processor:
            yield processor

    @property
    def edge_detector(self) -> EdgeDetector:
        """
        Gets the edge detector.
        """
        if not hasattr(self, "_edge_detector"):
            from enfugue.diffusion.support.edge import EdgeDetector
            self._edge_detector = EdgeDetector(
                self.model_dir,
                device=self.device,
                dtype=self.dtype,
                offline=self.offline
            )
            self._edge_detector.task_callback = self.task_callback
        return self._edge_detector

    @property
    def line_detector(self) -> LineDetector:
        """
        Gets the line detector.
        """
        if not hasattr(self, "_line_detector"):
            from enfugue.diffusion.support.line import LineDetector
            self._line_detector = LineDetector(
                self.model_dir,
                device=self.device,
                dtype=self.dtype,
                offline=self.offline
            )
            self._line_detector.task_callback = self.task_callback
        return self._line_detector

    @property
    def depth_detector(self) -> DepthDetector:
        """
        Gets the depth detector.
        """
        if not hasattr(self, "_depth_detector"):
            from enfugue.diffusion.support.depth import DepthDetector
            self._depth_detector = DepthDetector(
                self.model_dir,
                device=self.device, 
                dtype=self.dtype,
                offline=self.offline
            )
            self._depth_detector.task_callback = self.task_callback
        return self._depth_detector

    @property
    def pose_detector(self) -> PoseDetector:
        """
        Gets the pose detector.
        """
        if not hasattr(self, "_pose_detector"):
            from enfugue.diffusion.support.pose import PoseDetector
            self._pose_detector = PoseDetector(
                self.model_dir,
                device=self.device,
                dtype=self.dtype,
                offline=self.offline
            )
            self._pose_detector.task_callback = self.task_callback
        return self._pose_detector

    def __call__(self, controlnet: CONTROLNET_LITERAL, image: Image) -> Image:
        """
        A shorthand for executing with a single processor.
        """
        with self.processor(controlnet) as process:
            return process(image)
