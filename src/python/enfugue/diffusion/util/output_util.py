from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from dataclasses import dataclass
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["EnfugueStableDiffusionPipelineOutput"]

@dataclass
class EnfugueStableDiffusionPipelineOutput(StableDiffusionPipelineOutput):
    """
    Adds an optional video output to the pipeline output.
    """
    video: Optional[List[Image]] = None
