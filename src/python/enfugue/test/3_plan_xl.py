"""
Uses the engine to create a simple image using default settings
"""
import os
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.plan import DiffusionPlan, DiffusionNode, DiffusionStep
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.constants import DEFAULT_SDXL_MODEL, DEFAULT_SDXL_REFINER

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        kwargs = {
            "size": 1024,
            "model": DEFAULT_SDXL_MODEL,
            "refiner": DEFAULT_SDXL_REFINER,
            "prompt": "A happy looking puppy",
            "upscale_diffusion_guidance_scale": 5.0,
            "upscale_diffusion_strength": 0.3,
            "upscale_diffusion_steps": 50
        }

        manager = DiffusionPipelineManager()
        
        # Base plan
        manager.seed = 123456
        plan = DiffusionPlan.assemble(**kwargs)
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-xl.png"))

        # Upscale
        plan.upscale_steps = [{
            "amount": 2,
            "method": "esrgan"
        }]
        manager.seed = 123456
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-xl-upscale.png"))

        # Upscale diffusion at 2x
        plan.upscale_steps = [{
            "amount": 2,
            "method": "esrgan",
            "strength": 0.2,
            "num_inference_steps": 50,
            "guidance_scale": 10.0
        }]
        plan.upscale_diffusion = True
        manager.seed = 123456
        result = plan.execute(manager)["images"][0]
        result.save(os.path.join(save_dir, "./puppy-plan-xl-upscale-diffusion.png"))
        
        # Upscale again alone
        plan = DiffusionPlan.upscale_image(
            image=result,
            upscale_steps=[{
                "amount": 2,
                "method": "esrgan",
                "strength": 0.2,
                "chunking_size": 512,
                "strength": 0.2,
                "num_inference_steps": 50,
                "guidance_scale": 10.0
            }],
            **kwargs
        )
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-xl-upscale-solo.png"))

if __name__ == "__main__":
    main()
