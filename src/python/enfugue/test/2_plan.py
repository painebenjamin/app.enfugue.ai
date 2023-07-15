"""
Uses the engine to create a simple image using default settings
"""
import os
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.manager import DiffusionPipelineManager

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        manager = DiffusionPipelineManager()
        
        # Base plan
        manager.seed = 123456
        plan = DiffusionPlan.from_nodes(prompt="A happy looking puppy", upscale_diffusion_guidance_scale=10.0)
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan.png"))

        # Inpainting + region prompt + background removal
        plan = DiffusionPlan.from_nodes(
            prompt="A cat and dog laying on a couch",
            nodes=[
                {
                    "x": 0,
                    "y": 128,
                    "w": 256,
                    "h": 256,
                    "prompt": "A golden retriever laying down",
                    "remove_background": True
                },
                {
                    "x": 256,
                    "y": 128,
                    "w": 256,
                    "h": 256,
                    "prompt": "A cat laying down",
                    "remove_background": True
                }
            ]
        )
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-kitty-inpaint.png"))

        # Upscale
        plan.outscale = 2
        plan.upscale = "esrgan"
        manager.seed = 12345
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-upscale.png"))

        # Upscale diffusion
        plan.upscale_diffusion = True
        manager.seed = 12345
        result = plan.execute(manager)["images"][0]
        result.save(os.path.join(save_dir, "./puppy-plan-upscale-diffusion.png"))

        # Upscale again just from the image
        plan = DiffusionPlan.upscale_image(
            result,
            outscale=2,
            upscale="esrgan",
            upscale_diffusion=True,
            upscale_diffusion_guidance_scale=10.0,
            upscale_diffusion_strength=0.2,
            upscale_diffusion_steps=40,
            upscale_diffusion_chunking_size=256,
            upscale_diffusion_chunking_blur=256,
        )
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-upscale-solo.png"))


if __name__ == "__main__":
    main()
