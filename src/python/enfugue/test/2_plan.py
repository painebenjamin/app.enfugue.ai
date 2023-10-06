"""
Uses the engine to create a simple image using default settings
"""
import os
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.manager import DiffusionPipelineManager

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        manager = DiffusionPipelineManager()
        # Base plan
        manager.seed = 123456
        plan = DiffusionPlan.assemble(size=512, prompt="A happy looking puppy")
        """
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan.png"))

        # Inpainting + region prompt + background removal
        plan = DiffusionPlan.assemble(
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
        """
        # Upscale
        plan.upscale_steps = {
            "amount": 2,
            "method": "esrgan"
        }
        manager.seed = 12345
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-upscale.png"))

        # Upscale diffusion
        plan.upscale_steps = {
            "amount": 2,
            "method": "esrgan",
            "strength": 0.2
        }
        manager.seed = 12345
        result = plan.execute(manager)["images"][0]
        result.save(os.path.join(save_dir, "./puppy-plan-upscale-diffusion.png"))

        # Upscale again just from the image
        plan = DiffusionPlan.upscale_image(
            size=512,
            image=result,
            upscale_steps=[{
                "method": "esrgan",
                "amount": 2,
                "strength": 0.2,
                "chunking_size": 256,
            }]
        )
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-upscale-solo.png"))


if __name__ == "__main__":
    main()
