from __future__ import annotations

from typing import Iterator, Any, TYPE_CHECKING
from contextlib import contextmanager
from PIL import Image
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
        model: torch.nn.Module,
        **kwargs: Any
    ) -> None:
        super(MidasImageProcessor, self).__init__(**kwargs)
        self.model = model

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth prediction then returns to an image
        """
        size = image.size
        result = self.model(
            image,
            output_type = "pil",
            image_resolution=max(size),
        )
        result = result.resize(size)
        return result

class NormalImageProcessor(MidasImageProcessor):
    """
    Extends the depth processor to perform normal estimation
    """
    @staticmethod
    def from_midas(midas_processor: MidasImageProcessor) -> NormalImageProcessor:
        """
        Create this from a midas processor
        """
        return NormalImageProcessor(model=midas_processor.model)

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth prediction then transforms into a normal
        """
        size = image.size
        _, result = self.model(
            image,
            output_type = "pil",
            image_resolution=max(size),
            depth_and_normal=True
        )
        result = result.resize(size)
        return result

class DepthDetector(SupportModel):
    """
    Uses MiDaS v2 to predict depth.
    Uses depth prediction to generate normal maps.
    """

    MIDAS_MODEL_TYPE = "dpt_hybrid"
    MIDAS_MODEL_URL = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"

    @contextmanager
    def midas(self) -> Iterator[MidasImageProcessor]:
        """
        Executes MiDaS depth estimation
        """
        import torch

        with self.context():
            from enfugue.diffusion.support.depth.midas import MidasDetector # type: ignore
            model = MidasDetector.from_pretrained(
                self.get_model_file(self.MIDAS_MODEL_URL),
                model_type=self.MIDAS_MODEL_TYPE
            )
            model = model.to(self.device)
            processor = MidasImageProcessor(model)
            yield processor
            del model
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
