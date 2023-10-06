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

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(HERE, "test-results", "inpaint")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        manager = DiffusionPipelineManager()
        manager.safe = False

        image = PIL.Image.open(os.path.join(HERE, "test-images", "small-inpaint.jpg"))
        mask = PIL.Image.open(os.path.join(HERE, "test-images", "small-inpaint-mask-invert.jpg"))
        prompt = "a man breakdancing in front of a bright blue sky"
        negative_prompt = "tree, skyline, buildings"
        
        plan = DiffusionPlan.assemble(
            size=512,
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = 20,
            image = image,
            mask = mask,
            invert_mask = True
        )

        plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"result.png"))
        
        plan = DiffusionPlan.assemble(
            size=512,
            inpainter = "v1-5-pruned.ckpt", # Force 4-dim inpainting
            prompt = "blue sky and green grass",
            negative_prompt = negative_prompt,
            num_inference_steps = 20,
            image = image,
            mask = mask,
            invert_mask = True
        )
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"result-4-dim.png"))
        

if __name__ == "__main__":
    main()
