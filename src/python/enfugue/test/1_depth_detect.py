import io
import os
import PIL
import requests

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.util import image_from_uri

from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://github.com/pytorch/hub/raw/master/images/classification.jpg"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "depth-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = image_from_uri(BASE_IMAGE)
        manager = DiffusionPipelineManager()
        controlnets = ["depth", "normal"]
        image.save(os.path.join(save_dir, "base.png"))
        with manager.control_image_processor.processors(*controlnets) as processors:
            for controlnet, processor in zip(controlnets, processors):
                processor(image).save(os.path.join(save_dir, f"detect-{controlnet}.png"))

if __name__ == "__main__":
    main()
