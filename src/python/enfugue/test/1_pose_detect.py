"""
Uses the pipemanager to create a simple image using default settings
"""
import io
import os
import PIL
import requests
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/input.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "pose-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image = PIL.Image.open(io.BytesIO(requests.get(BASE_IMAGE, stream=True).content))
        manager = DiffusionPipelineManager()
        manager.pose_detector.detect(image).save(os.path.join(save_dir, "detect-pose.png"))

if __name__ == "__main__":
    main()
