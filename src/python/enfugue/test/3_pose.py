"""
Tests automatic loading of controlnet and pose detection
"""
import os
import PIL

from enfugue.util import logger
from enfugue.diffusion.engine import DiffusionEngine
from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "pose-detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            base_image = engine(prompt="A chef waving hello", seed=12345)["images"][0]
            base_image.save(os.path.join(output_dir, "base.png"))
            output_path = os.path.join(output_dir, f"pose.png")
            engine(
                seed=12345,
                prompt="A schoolteacher waving hello",
                controlnet="pose",
                control_image=base_image
            )["images"][0].save(output_path)
            logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
