"""
Tests automatic loading of motion module/animator pipeline
"""
import os
import torch

from datetime import datetime

from typing import Literal, Optional

from enfugue.util import logger, fit_image, profiler, image_from_uri
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.constants import *
from enfugue.diffusion.util import Video, GridMaker

from PIL import Image

from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "dragnuwa")
    input_dir = os.path.join(here, "test-images")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        manager = DiffusionPipelineManager()
        with manager.drag_animator.nuwa() as drag_nuwa:
            input_image = image_from_uri(os.path.join(input_dir, "nuwa.png"))
            frames = drag_nuwa(
                input_image,
                motion_vectors=[
                    [
                        {
                            "anchor": (x1, y1)
                        },
                        {
                            "anchor": (x2, y2)
                        }
                    ]
                    for ((x1, y1), (x2, y2)) in [
                        ((222, 55), (0, 0)),
                        ((519, 21), (575, 0)),
                        ((205, 197), (0, 319)),
                        ((527, 180), (575, 319))
                    ]
                ]
            )[0]
            Video(frames).save(
                os.path.join(output_dir, "output.gif"),
                overwrite=True,
                rate=8.0
            )

if __name__ == "__main__":
    main()
