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
from enfugue.diffusion.manager import DiffusionPipelineManager

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = {
    "medium": ("a smiling woman with white teeth", "watermark"),
    "large": ("A beautiful mountain view, rolling hills and clouds", "power lines, signs, posters")
}

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(HERE, "test-results", "crop-inpaint")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        manager = DiffusionPipelineManager()
        manager.safe = False

        for size in ["medium", "large"]:
            image = PIL.Image.open(os.path.join(HERE, "test-images", f"{size}-inpaint.jpg"))
            mask = PIL.Image.open(os.path.join(HERE, "test-images", f"{size}-inpaint-mask.jpg"))
            width, height = image.size
            prompt, negative_prompt = PROMPTS[size]
            
            plan = LayeredInvocation.assemble(
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = 20,
                width = width,
                height = height,
                mask=mask,
                image=image,
                strength=1.0,
            )

            plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"{size}-result.png"))

if __name__ == "__main__":
    main()
