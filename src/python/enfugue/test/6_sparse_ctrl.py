"""
Tests automatic loading of motion module/animator pipeline
"""
import os
import PIL

from enfugue.util import logger, fit_image, profiler
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.constants import *
from enfugue.diffusion.util import Video

from PIL import Image

from pibble.util.log import DebugUnifiedLoggingContext

PROMPT = "a beautiful woman smiling, open mouth, bright teeth"
FRAMES = 16
RATE = 8.0

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "animation")
    input_dir = os.path.join(here, "test-images")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_1 = Image.open(os.path.join(input_dir, "one.png"))
    image_2 = Image.open(os.path.join(input_dir, "two.png"))

    with DebugUnifiedLoggingContext():
        import torch
        manager = DiffusionPipelineManager()
        manager.seed = 12345
        manager.lora = [("v3_sd15_adapter.ckpt", 1.0)]
        manager.model = "epicrealism_pureEvolutionV5.safetensors"
        pipeline = manager.animator_pipeline
        pipeline.controlnets = {
            "sparse-rgb": pipeline.get_sparse_controlnet(
                "sparse-rgb",
                cache_dir=manager.engine_cache_dir
            ).to(torch.float16)
        }
        frames = pipeline(
            prompt="a boy is playing outside",
            width=512,
            height=768,
            animation_frames=16,
            control_images={
                "sparse-rgb": [
                    {"image": image_1, "frame": 0},
                    {"image": image_2, "frame": 15}
                ]
            },
            device=manager.device
        )["images"]

        Video(frames).save(
            os.path.join(output_dir, "interpolation.mp4"),
            rate=8.0,
            overwrite=True
        )
if __name__ == "__main__":
    main()
