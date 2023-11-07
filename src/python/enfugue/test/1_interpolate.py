import io
import os
import PIL
import requests

from enfugue.util import image_from_uri, fit_image
from datetime import datetime

from pibble.util.log import DebugUnifiedLoggingContext

BASE_IMAGE = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "background-remove")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = image_from_uri(BASE_IMAGE)
        image = fit_image(image, width=544, height=512, fit="cover")
        left = fit_image(image, width=512, height=512, anchor="top-left")
        right = fit_image(image, width=512, height=512, anchor="top-right")
        left.save(os.path.join(save_dir, "left.png"))
        right.save(os.path.join(save_dir, "right.png"))

        #manager = DiffusionPipelineManager()
        frames = 32
        from enfugue.diffusion.support.interpolate.interpolator import Interpolator
        interpolator = Interpolator()
        with interpolator.interpolate() as interpolate:
            start = datetime.now()
            frame_list = interpolate(
                left,
                right,
                [
                    (i+1)/(frames-1)
                    for i in range(frames-2)
                ]
            )
            for i, frame in enumerate(frame_list):
                frame.save(os.path.join(save_dir, f"interpolated-{i+1}.png"))
            seconds = (datetime.now() - start).total_seconds()
            average = seconds / (frames-2)
            print(f"Interpolated {frames-2} frames in {seconds} ({average}s/frame)")

if __name__ == "__main__":
    main()
