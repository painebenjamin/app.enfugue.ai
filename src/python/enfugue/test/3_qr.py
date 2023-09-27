"""
Tests automatic loading of controlnet and edge detection
"""
import os
import PIL

from enfugue.util import logger, image_from_uri
from enfugue.diffusion.engine import DiffusionEngine
from pibble.util.log import DebugUnifiedLoggingContext

IMAGE = "https://1.bp.blogspot.com/-dHN4KiD3dsU/XRxU5JRV7DI/AAAAAAAAAz4/u1ynpCMIuKwZMA642dHEoXFVKuHQbJvwgCEwYBhgL/s1600/qr-code.png"

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "qr-code")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            image = image_from_uri(IMAGE)
            image = image.resize((512, 512), resample=PIL.Image.Resampling.NEAREST)
            image.save(os.path.join(output_dir, "base.png"))
            output_path = os.path.join(output_dir, f"qr-code.png")
            engine(
                seed=54321,
                prompt="A creeping vine crawling up a trellice",
                control_images=[{
                    "controlnet": "qr",
                    "image": image
                }],
            )["images"][0].save(output_path)
            logger.info(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
