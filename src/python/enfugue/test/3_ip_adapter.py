"""
Tests automatic loading of ip adapter
"""
import os
import PIL

from enfugue.util import logger, image_from_uri
from enfugue.diffusion.engine import DiffusionEngine
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "ip-adapter")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            base_image = image_from_uri(BASE_IMAGE)
            base_image.save(os.path.join(output_dir, "base.png"))
            output_path = os.path.join(output_dir, f"sd15.png")
            engine(
                seed=12345,
                prompt="A basketball game",
                image=base_image,
                ip_adapter_scale=0.5
            )["images"][0].save(output_path)
            logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
