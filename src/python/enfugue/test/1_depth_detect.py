"""
Uses the pipemanager to create a simple image using default settings
"""
import io
import os
import PIL
import requests
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "depth-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image = PIL.Image.open(io.BytesIO(requests.get(BASE_IMAGE, stream=True).content))
        manager = DiffusionPipelineManager()
        manager.depth_detector.midas(image).save(os.path.join(save_dir, "depth-detect.png"))
        manager.depth_detector.normal(image).save(os.path.join(save_dir, "normal-detect.png"))

if __name__ == "__main__":
    main()
