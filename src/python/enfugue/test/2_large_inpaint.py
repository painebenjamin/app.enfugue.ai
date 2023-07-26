"""
Uses the engine to create a simple image using default settings
"""
import os
import PIL
import traceback

from typing import List
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.manager import DiffusionPipelineManager

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(HERE, "test-images", "large-inpaint.jpg")
MASK = os.path.join(HERE, "test-images", "large-inpaint-mask.jpg")

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(HERE, "test-results", "crop-inpaint")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        manager = DiffusionPipelineManager()
        manager.safe = False

        image = PIL.Image.open(IMAGE)
        mask = PIL.Image.open(MASK)
        width, height = image.size
        
        plan = DiffusionPlan.from_nodes(
            prompt = "A beautiful mountain view, rolling hills and clouds",
            negative_prompt = "power lines, signs, posters",
            num_inference_steps = 20,
            width = width,
            height = height,
            nodes = [
                {
                    "image": image,
                    "mask": mask,
                    "inpaint": True,
                }
            ]
        )

        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "result.png"))

if __name__ == "__main__":
    main()
