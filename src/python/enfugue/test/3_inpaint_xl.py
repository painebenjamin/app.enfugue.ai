"""
Uses the engine to create a simple image using default settings
"""
import os
import PIL
import traceback

from typing import List
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.constants import DEFAULT_SDXL_MODEL
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
        
        plan = LayeredInvocation.assemble(
            size = 1024,
            prompt = prompt,
            model = DEFAULT_SDXL_MODEL,
            num_inference_steps = 20,
            image = image,
            mask = mask
        )

        plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"result-xl.png"))

        # Force 4-dim
        plan = LayeredInvocation.assemble(
            size = 1024,
            prompt = prompt,
            inpainter = DEFAULT_SDXL_MODEL,
            num_inference_steps = 20,
            image = image,
            mask = mask
        )

        plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"result-xl-4dim.png"))

if __name__ == "__main__":
    main()
