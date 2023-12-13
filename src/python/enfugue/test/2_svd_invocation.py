"""
Uses the engine to exercise SVD plan instantiation and execution
"""
import os
import time
from pibble.util.log import DebugUnifiedLoggingContext

from enfugue.util import image_from_uri, fit_image, logger

from enfugue.diffusion.invocation import StableVideoDiffusionInvocation
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import Video

def format_rate(rate: float) -> str:
    unit = "it/s"
    if rate < 1.0:
        unit = "s/it"
        rate = 1.0 / rate
    return f"{rate:.2f} {unit}"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "svd")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = image_from_uri("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
        image = fit_image(image, width=1024, height=576, fit="cover", anchor="center-center").convert("RGB")

        manager = DiffusionPipelineManager()

        plan = StableVideoDiffusionInvocation.assemble(
            image=image,
            model="svd",
            seed=12345
        )
        frames = plan.execute(
            manager,
            task_callback=lambda arg: logger.info(arg),
        )["images"]
        Video(frames).save(os.path.join(save_dir, "svd.mp4"), rate=7.0)

        plan.model = "svd_xt"
        plan.interpolate_frames = 8
        plan.reflect = True
        frames = plan.execute(
            manager,
            task_callback=lambda arg: logger.info(arg),
            progress_callback=lambda step, total, rate: logger.info(f"{step:02d}/{total:02d} @ {format_rate(rate)}")
        )["frames"]
        Video(frames).save(os.path.join(save_dir, "xt.mp4"), rate=60, overwrite=True)

if __name__ == "__main__":
    main()
