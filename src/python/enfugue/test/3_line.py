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
    output_dir = os.path.join(here, "test-results", "line-detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            base_image = engine(prompt = "A suburban home frontage", seed=12345)["images"][0]
            base_image.save(os.path.join(output_dir, "base.png"))
            for controlnet in ["line", "anime", "mlsd"]:
                output_path = os.path.join(output_dir, f"{controlnet}.png")
                engine(
                    seed=54321,
                    prompt="A suburban home frontage",
                    control_images=[{
                        "controlnet": controlnet,
                        "image": base_image
                    }],
                )["images"][0].save(output_path)
                logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
