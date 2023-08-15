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
            prompt = prompt,
            model = "sd_xl_base_1.0.safetensors",
            num_inference_steps = 50,
            image = image,
            mask = mask
        )

        plan.execute(manager)["images"][0].save(os.path.join(save_dir, f"result-xl.png"))
        
        ## Now run base to compare
        from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLInpaintPipeline
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            os.path.join(manager.engine_diffusers_dir, "sd_xl_base_1.0")
        )
        pipeline.to("cuda")
        pipeline(
            prompt = prompt,
            image = image,
            mask_image = mask,
        )["images"][0].save(os.path.join(save_dir, f"result-xl-compare.png"))

if __name__ == "__main__":
    main()
