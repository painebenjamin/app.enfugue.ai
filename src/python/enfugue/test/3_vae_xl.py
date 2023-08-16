"""
Uses the engine to create a simple image using default settings
"""
import os
import PIL
import traceback

from typing import List
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.constants import DEFAULT_SDXL_MODEL, DEFAULT_SDXL_REFINER

VAE = [
    "xl",
    "xl16",
]

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "vae-xl")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        manager.safe = False
        manager.model = DEFAULT_SDXL_MODEL
        kwargs = {
            "prompt": "A happy-looking puppy",
            "num_inference_steps": 20
        }

        def run_and_save(filename: str) -> None:
            manager.seed = 1234567
            manager(**kwargs)["images"][0].save(os.path.join(save_dir, filename))
        
        run_and_save("original.png")
        manager.refiner = DEFAULT_SDXL_REFINER
        run_and_save("original-refined.png")
        for vae in VAE:
            try:
                manager.refiner = None
                manager.vae = vae
                run_and_save(f"{vae}.png")
                manager.refiner = DEFAULT_SDXL_REFINER
                run_and_save(f"{vae}-refined.png")
            except Exception as ex:
                logger.error("Error with VAE {0}: {1}({2})".format(vae, type(ex).__name__, ex))
                logger.info(traceback.format_exc())
        


if __name__ == "__main__":
    main()
