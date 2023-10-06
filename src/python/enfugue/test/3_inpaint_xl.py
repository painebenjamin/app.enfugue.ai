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
        
        image = PIL.Image.open(os.path.join(HERE, "test-images", "inpaint-xl.jpg"))
        mask = PIL.Image.open(os.path.join(HERE, "test-images", "inpaint-xl-mask.jpg"))
        
        manager = DiffusionPipelineManager()
        manager.safe = False

        prompt = "a huge cactus standing in the desert"
        
        plan = DiffusionPlan.assemble(
            size = 1024,
            prompt = prompt,
            model = "sd_xl_base_1.0.safetensors",
            num_inference_steps = 50,
            image = image,
            mask = mask
        )

        plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"result-xl.png"))

if __name__ == "__main__":
    main()
