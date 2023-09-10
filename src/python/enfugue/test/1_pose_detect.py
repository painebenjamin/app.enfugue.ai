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
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "pose-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        images = [
            image_from_uri(BASE_IMAGE)
            for BASE_IMAGE in BASE_IMAGES
        ]
        manager = DiffusionPipelineManager()
        controlnets = ["openpose", "dwpose"]

        with manager.control_image_processor.pose_detector.openpose() as openpose:
            for i, image in enumerate(images):
                image.save(os.path.join(save_dir, f"base-{i}.png"))
                start_openpose = datetime.now()
                openpose(image).save(os.path.join(save_dir, f"detect-openpose-{i}.png"))
                openpose_time = (datetime.now() - start_openpose).total_seconds()
                logger.info(f"Openpose took {openpose_time:0.03f}")
        try:
            with manager.control_image_processor.pose_detector.dwpose() as dwpose:
                for i, image in enumerate(images):
                    start_dwpose = datetime.now()
                    dwpose(image).save(os.path.join(save_dir, f"detect-dwpose-{i}.png"))
                    dwpose_time = (datetime.now() - start_dwpose).total_seconds()
                    logger.info(f"dwpose took {dwpose_time:0.03f}")
        except Exception as ex:
            logger.warning(f"Received exception using DWPose: {type(ex).__name__}({ex})")

if __name__ == "__main__":
    main()
