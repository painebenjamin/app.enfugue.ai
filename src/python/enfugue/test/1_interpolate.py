import io
import os
import PIL
import requests

from enfugue.util import image_from_uri, fit_image
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import Video, interpolate_frames

from datetime import datetime

from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "interpolate")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = image_from_uri(BASE_IMAGE)
        image = fit_image(image, width=544, height=512, fit="cover")

        left = fit_image(image, width=512, height=512, anchor="top-left")
        right = fit_image(image, width=512, height=512, anchor="top-right")

        left.save(os.path.join(save_dir, "left.png"))
        right.save(os.path.join(save_dir, "right.png"))

        manager = DiffusionPipelineManager()
        with manager.interpolator.film() as interpolate:
            start = datetime.now()
            images = interpolate(left, right, num_frames=10, include_ends=True)
            seconds = (datetime.now() - start).total_seconds()
            Video(images).save(os.path.join(save_dir, "interpolated.gif"), rate=12.0, overwrite=True)
            frames = len(images) - 2
            average = seconds / frames

            print(f"Interpolated {frames} frames in {seconds} ({average}s/frame)")

            def progress_callback(step: int, total: int, rate: float) -> None:
                print(f"Interpolated {total} frames at {rate}s/frame")

            frames = interpolate_frames(
                frames=[left, right],
                multiplier=118,
                interpolate=interpolate,
                progress_callback=progress_callback
            )

            Video(frames).save(os.path.join(save_dir, "interpolated-long.mp4"), rate=60.0, overwrite=True)

if __name__ == "__main__":
    main()
