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
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "face")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        images = [
            image_from_uri(BASE_IMAGE)
            for BASE_IMAGE in BASE_IMAGES
        ]

        manager = DiffusionPipelineManager()
        images.append(
            manager(
                prompt="A middle-aged man"
            )["images"][0]
        )

        for i, image in enumerate(images):
            image.save(os.path.join(save_dir, f"base-{i}.png"))

        with manager.upscaler.face_restore() as restore:
            for i, image in enumerate(images):
                restore(image).save(os.path.join(save_dir, f"restored-{i}.png"))

if __name__ == "__main__":
    main()
