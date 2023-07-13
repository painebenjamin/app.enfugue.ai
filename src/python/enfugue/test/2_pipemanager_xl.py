"""
Uses the pipemanager to create a simple image using default settings
"""
import os
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        kwargs = {
            "prompt": "A happy looking puppy",
            "guidance_scale": 5.0,
            "width": 1024,
            "height": 1024
        }

        # Start with absolute defaults.
        # Even if there's nothing on your machine, this should work by downloading everything needed.
        manager = DiffusionPipelineManager()
        manager.size = 1024
        manager.model = "sd_xl_base_0.9.safetensors"
        
        def run_and_save(filename: str) -> None:
            manager.seed = 1238421 # set seed for reproduceability
            manager(**kwargs)["images"][0].save(os.path.join(save_dir, filename))

        run_and_save("puppy-xl.png")

        # Add the refiner
        manager.refiner = "sd_xl_refiner_0.9.safetensors"
        run_and_save("puppy-xl-refined.png")
        

if __name__ == "__main__":
    main()
