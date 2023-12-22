"""
Tests automatic loading of ip adapter
"""
import os
import PIL

from enfugue.util import logger, image_from_uri
from enfugue.diffusion.constants import *
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "ip-adapter")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        manager = DiffusionPipelineManager()
        logger.info(manager.ip_adapter.face_analyzer)
        base_image = image_from_uri(BASE_IMAGE)

if __name__ == "__main__":
    main()
