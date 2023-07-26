import io
import os
import PIL
import requests
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://github.com/pytorch/hub/raw/master/images/ssd.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "line-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = PIL.Image.open(io.BytesIO(requests.get(BASE_IMAGE, stream=True).content))
        manager = DiffusionPipelineManager()
        manager.line_detector.detect(image).save(os.path.join(save_dir, "detect-line.png"))
        manager.line_detector.detect(image, anime=True).save(os.path.join(save_dir, "detect-anime.png"))
        manager.line_detector.mlsd(image).save(os.path.join(save_dir, "detect-mlsd.png"))

if __name__ == "__main__":
    main()
