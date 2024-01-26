import io
import os
import PIL
import requests
from datetime import datetime

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import debug_tensors, get_vram_info
from enfugue.util import image_from_uri, logger

from pibble.util.numeric import human_size
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGES = [
    "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png",
    "https://raw.githubusercontent.com/IDEA-Research/DWPose/onnx/ControlNet-v1-1-nightly/test_imgs/girls.jpg",
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

        for i, image in enumerate(images):
            image.save(os.path.join(save_dir, f"base-{i}.png"))

        for i in range(2):
            free, total = get_vram_info()
            logger.info("Starting at {0} free.".format(human_size(free)))
            with manager.face_analyzer.insightface() as detect:
                new_free, total = get_vram_info()
                logger.info("{0} after initializing.".format(human_size(new_free)))
                loaded = free - new_free
                logger.info("Used {0} to load.".format(
                    human_size(loaded)
                ))
                for i, image in enumerate(images):
                    debug_tensors(embeddings=detect(image))
                new_free, total = get_vram_info()
                loaded = free - new_free
                logger.info("Used {0} after inference.".format(
                    human_size(loaded)
                ))

            new_free, total = get_vram_info()
            loaded = free - new_free
            logger.info("Used {0} after cleaning.".format(
                human_size(loaded)
            ))
            import time
            time.sleep(2)

if __name__ == "__main__":
    main()
