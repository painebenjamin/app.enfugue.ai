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
        
        def run_and_save(filename: str) -> None:
            manager.seed = 1238421 # set seed for reproduceability
            manager(**kwargs)["images"][0].save(os.path.join(save_dir, filename))

        manager.chunking_size = 128
        manager.chunking_blur = 128
        manager.safe = False
        
        # Run once with sd 1.5
        #run_and_save("puppy.png")

        # Now switch to SDXL.
        # Default size in SDXL is 1024, this will catch that default
        manager.model = "sd_xl_base_0.9.safetensors"
        #run_and_save("puppy-xl.png")

        # Add the refiner
        manager.refiner = "sd_xl_refiner_0.9.safetensors"
        #run_and_save("puppy-xl-refined.png")
        
        kwargs["prompt"] = "a photograph of mike tyson after winning a fight, tired sweaty and beaten, fighter afterbattle, highly detailed, intricate detail, 4k, 8k, sharp focus, ultra-detailed, portrait photography, cinestill 800t, Fujifilm XT3"
        kwargs["negative_prompt"] = "deformed, (((distorted face))), ((malformed face)), blurry, bad anatomy, bad eyes, disfigured, mutation, mutated, extra limb, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, out of focus, long neck, long body, mutated hands and fingers, out of frame, watermark, cut off, bad art, grainy"

        run_and_save("test.png")
        # Try something even bigger
        #kwargs["width"] = 1920
        #kwargs["height"] = 1080
        #run_and_save("puppy-xxl-refined.png")


if __name__ == "__main__":
    main()
