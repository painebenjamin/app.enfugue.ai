"""
Tests automatic loading of controlnet and edge detection
"""
import os
import PIL

from enfugue.util import logger
from enfugue.diffusion.engine import DiffusionEngine
from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-images", "edge-detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine() as engine:
            base_image = engine(prompt = "A suburban home frontage", seed=12345)["images"][0]
            base_image.save(os.path.join(output_dir, "base.png"))
            for controlnet in ["mlsd", "hed", "canny"]:
                output_path = os.path.join(output_dir, f"{controlnet}.png")
                engine(
                    seed=12345,
                    prompt="A suburban home frontage",
                    controlnet=controlnet,
                    control_image=base_image
                )["images"][0].save(output_path)
                logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
