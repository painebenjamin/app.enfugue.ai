"""
Uses the engine to create a simple image using default settings
"""
import os
import PIL
from typing import List
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.manager import DiffusionPipelineManager

SCHEDULERS = [
    "ddim",
    "ddpm",
    "deis",
    "dpmsm",
    "dpmss",
    "heun",
    "dpmd",
    "dpmsde",
    "unipc",
    "lmsd",
    "pndm",
    "eds",
    "eads"
]

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "scheduler")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        kwargs = {
            "prompt": "A happy-looking puppy",
            "num_inference_steps": 10,
            "latent_callback_steps": 1,
            "latent_callback_type": "pil"
        }

        def run_and_save(filename: str) -> None:
            steps = kwargs["num_inference_steps"]
            basename, ext = os.path.splitext(filename)
            j = 0
            def intermediate_callback(images: List[PIL.Image.Image]) -> None:
                nonlocal j
                images[0].save(os.path.join(save_dir, f"{basename}-{j:02d}{ext}"))
                j += 1
            kwargs["latent_callback"] = intermediate_callback
            manager.seed = 1234567
            manager(**kwargs)["images"][0].save(os.path.join(save_dir, f"{basename}-{steps:02d}{ext}"))
        
        for scheduler in SCHEDULERS:
            try:
                manager.scheduler = scheduler
                run_and_save(f"puppy-{scheduler}.png")
            except Exception as ex:
                logger.error("Error with scheduler {0}: {1}({2})".format(scheduler, type(ex).__name__, ex))

if __name__ == "__main__":
    main()
