"""
Tests automatic loading of controlnet and edge detection
"""
import os
import PIL

from enfugue.util import logger
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.diffusion.constants import *
from pibble.util.log import DebugUnifiedLoggingContext

CONTROLNETS = ["canny"]

def main() -> None:
    prompt = "A suburban home frontage"
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "edge-detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            base_image = engine(
                prompt=prompt,
                model=DEFAULT_SDXL_MODEL,
                seed=12345,
                num_inference_steps=25
            )["images"][0]
            base_image.save(os.path.join(output_dir, "base-xl.png"))
            for controlnet in CONTROLNETS:
                output_path = os.path.join(output_dir, f"{controlnet}-xl.png")
                engine(
                    seed=54321,
                    prompt=prompt,
                    model=DEFAULT_SDXL_MODEL,
                    controlnet=controlnet,
                    num_inference_steps=25,
                    control_image=base_image,
                    conditioning_scale=0.5,
                )["images"][0].save(output_path)
                logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
