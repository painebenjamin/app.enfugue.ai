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

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def run_test(
    input_file: str,
    output_file: str,
    frame_divisor: int,
    prompt: str,
    manager: DiffusionPipelineManager,
    controlnet: Literal["sparse-rgb", "sparse-scribble"],
    use_img2img: bool = False,
    conditioning_start: float = 0.0,
    conditioning_end: Optional[float] = None,
) -> None:
    frames = [
        frame for i, frame in
        enumerate(Video.file_to_frames(os.path.join(input_file)))
        if i % frame_divisor == 0
    ][:16]
    width, height = frames[0].size
    grid_maker = GridMaker(
        grid_columns=2,
        grid_size=512,
        use_video=True
    )
    grid_frames = [(
        {},
        "Original",
        frames,
        0.0
    )]
    for reduce_rate in [2, 3, 5, 7, 15]:
        start = datetime.now()
        invocation = LayeredInvocation.assemble(
            seed=123456,
            model="dreamshaper_8.safetensors",
            image=None if not use_img2img else frames,
            prompt=prompt,
            width=width,
            height=height,
            animation_frames=16,
            frame_decode_chunk_size=4,
            num_inference_steps=25,
            layers=[
                {
                    "image": image,
                    "frame": frame,
                    "control_units": [{
                        "controlnet": controlnet,
                        "start": conditioning_start,
                        "end": conditioning_end,
                    }]
                }
                for frame, image in enumerate(frames)
                if frame % reduce_rate == 0
            ],
        )
        frames = invocation.execute(manager)["images"]
        duration = (datetime.now() - start).total_seconds()
        grid_frames.append((
            {},
            f"Every {ordinal(reduce_rate)} Frame",
            frames,
            duration
        ))
    Video(grid_maker.collage(grid_frames)).save(
        output_file,
        rate=8.0,
        overwrite=True
    )

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "animation")
    input_dir = os.path.join(here, "test-images")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        manager = DiffusionPipelineManager()
        run_test(
            prompt="a dolphin swimming in the ocean",
            frame_divisor=6,
            input_file=os.path.join(input_dir, "pexels-dolphin.mp4"),
            output_file=os.path.join(output_dir, "dolphin.mp4"),
            controlnet="sparse-scribble",
            manager=manager
        )
        """
        run_test(
            prompt="a man dressed at santa clause is dancing",
            frame_divisor=3,
            input_file=os.path.join(input_dir, "pexels-santa.mp4"),
            output_file=os.path.join(output_dir, "santa.mp4"),
            controlnet="sparse-rgb",
            conditioning_start=0.0,
            conditioning_end=0.60,
            manager=manager
        )
        """
if __name__ == "__main__":
    main()
