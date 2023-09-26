import io
import os
import PIL
import requests

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.util import image_from_uri

from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "background-remove")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = image_from_uri(BASE_IMAGE)
        manager = DiffusionPipelineManager()
        image.save(os.path.join(save_dir, "base.png"))
        from enfugue.util import remove_background
        remove_background(image)
        with manager.background_remover.remover() as remove:
            remove(image).save(os.path.join(save_dir, "removed.png"))

if __name__ == "__main__":
    main()
