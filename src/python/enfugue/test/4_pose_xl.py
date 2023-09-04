"""
Tests automatic loading of controlnet and pose detection
"""
import os
import PIL

from enfugue.util import logger
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.diffusion.constants import *
from pibble.util.log import DebugUnifiedLoggingContext

PROMPT = "A schoolteacher waving hello"
CONTROLNETS = ["pose"]

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "pose-detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            base_image = engine(
                model=DEFAULT_SDXL_MODEL,
                prompt=PROMPT,
                seed=12345
            )["images"][0]
            base_image.save(os.path.join(output_dir, "base-xl.png"))
            for controlnet in CONTROLNETS:
                output_path = os.path.join(output_dir, f"{controlnet}-xl.png")
                engine(
                    seed=54321,
                    model=DEFAULT_SDXL_MODEL,
                    prompt=PROMPT,
                    control_images=[{
                        "controlnet": controlnet,
                        "image": base_image,
                        "scale": 1.0
                    }]
                )["images"][0].save(output_path)
                logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
