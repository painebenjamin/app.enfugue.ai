import io
import os
import PIL
import requests
from datetime import datetime

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.util import image_from_uri, logger

from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGES = [
    "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png",
    "https://raw.githubusercontent.com/IDEA-Research/DWPose/onnx/ControlNet-v1-1-nightly/test_imgs/girls.jpg"
]

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "upscaling")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        images = [
            image_from_uri(BASE_IMAGE)
            for BASE_IMAGE in BASE_IMAGES
        ]
        manager = DiffusionPipelineManager()

        def time_function(function, name):
            start = datetime.now()
            result = function()
            time = (datetime.now() - start).total_seconds()
            logger.info(f"{name} took {time:.2f} seconds")
            return result

        with manager.upscaler.esrgan(tile=512) as esrgan:
            for i, image in enumerate(images):
                image.save(os.path.join(save_dir, f"base-{i}.png"))
                time_function(lambda: esrgan(image, outscale=2), f"2× ESRGAN Image {i}").save(
                    os.path.join(save_dir, f"esrgan-2x-{i}.png")
                )
                time_function(lambda: esrgan(image, outscale=4), f"4× ESRGAN Image {i}")

        with manager.upscaler.esrgan(tile=512, anime=True) as esrgan:
            for i, image in enumerate(images):
                time_function(lambda: esrgan(image, outscale=2), f"2× ESRGANime Image {i}").save(
                    os.path.join(save_dir, f"esrganime-2x-{i}.png")
                )
                time_function(lambda: esrgan(image, outscale=4), f"4× ESRGANime Image {i}")

        with manager.upscaler.gfpgan(tile=512) as gfpgan:
            for i, image in enumerate(images):
                time_function(lambda: esrgan(image, outscale=2), f"2× GFPGAN Image {i}").save(
                    os.path.join(save_dir, f"gfpgan-2x-{i}.png")
                )
                time_function(lambda: esrgan(image, outscale=4), f"4× GFPGAN Image {i}")

        with manager.upscaler.ccsr() as ccsr:
            time_function(lambda: ccsr(images[0], outscale=2), "2× CCSR Image").save(
                os.path.join(save_dir, "ccsr-2x.png")
            )
            time_function(lambda: ccsr(images[0], outscale=4), "4× CCSR Image").save(
                os.path.join(save_dir, "ccsr-4x.png")
            )

if __name__ == "__main__":
    main()
