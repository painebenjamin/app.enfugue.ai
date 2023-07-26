import io
import os
import PIL
import requests
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://github.com/pytorch/hub/raw/master/images/ssd.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "edge-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = PIL.Image.open(io.BytesIO(requests.get(BASE_IMAGE, stream=True).content))
        manager = DiffusionPipelineManager()
        manager.edge_detector.canny(image).save(os.path.join(save_dir, "detect-canny.png"))
        manager.edge_detector.hed(image).save(os.path.join(save_dir, "detect-hed.png"))
        manager.edge_detector.hed(image, scribble=True).save(os.path.join(save_dir, "detect-scribble.png"))
        manager.edge_detector.pidi(image).save(os.path.join(save_dir, "detect-pidi.png"))

if __name__ == "__main__":
    main()
