"""
Uses the engine to create a simple image using default settings
"""
import os
import PIL
import traceback
from typing import List, Any
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.manager import DiffusionPipelineManager

SCHEDULERS = [
    "ddim",
    "ddpm",
    "deis",
    "dpmsm",
    "dpmsmk",
    "dpmsmka",
    "dpmss",
    "dpmssk",
    "heun",
    "dpmd",
    "dpmdk",
    "adpmd",
    "adpmdk",
    "dpmsde",
    "unipc",
    "lmsd",
    "lmsdk",
    "pndm",
    "eds",
    "eads",
]

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "scheduler")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        manager.safe = False
        kwargs = {
            "prompt": "A happy-looking puppy",
            "num_inference_steps": 5,
            "latent_callback_steps": 1,
            "latent_callback_type": "pil"
        }
        multi_kwargs = {
            "width": 768,
            "chunking_size": 64,
        }

        def run_and_save(target_dir: str, **other_kwargs: Any) -> None:
            steps = kwargs["num_inference_steps"]
            j = 0
            def intermediate_callback(images: List[PIL.Image.Image]) -> None:
                nonlocal j
                images[0].save(os.path.join(target_dir, f"{j:02d}-intermediate.png"))
                j += 1
            kwargs["latent_callback"] = intermediate_callback
            manager.seed = 1234567
            manager(**{**kwargs, **other_kwargs})["images"][0].save(os.path.join(target_dir, f"{steps:02d}-final.png"))
        
        for scheduler in SCHEDULERS:
            target_dir = os.path.join(save_dir, scheduler)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            target_dir_single = os.path.join(target_dir, "single")
            if not os.path.exists(target_dir_single):
                os.makedirs(target_dir_single)
            target_dir_multi = os.path.join(target_dir, "multi")
            if not os.path.exists(target_dir_multi):
                os.makedirs(target_dir_multi)
            try:
                manager.scheduler = scheduler

                run_and_save(target_dir_single)
                run_and_save(target_dir_multi, **multi_kwargs)
            except Exception as ex:
                logger.error("Error with scheduler {0}: {1}({2})".format(scheduler, type(ex).__name__, ex))
                logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()
