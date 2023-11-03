"""
Uses the engine to create a simple image using default settings in XL
"""
import os
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.constants import DEFAULT_SDXL_MODEL, DEFAULT_SDXL_REFINER

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        kwargs = {
            "seed": 12345,
            "size": 1024,
            "model": DEFAULT_SDXL_MODEL,
            "prompt": "A happy looking puppy",
            "refiner": DEFAULT_SDXL_REFINER,
            "refiner_start": 0.85
        }

        manager = DiffusionPipelineManager()
        
        # Base plan
        plan = LayeredInvocation.assemble(**kwargs)
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-xl.png"))

        # Upscale
        plan.upscale = [{
            "amount": 2,
            "method": "esrgan"
        }]
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-xl-upscale.png"))

        # Upscale diffusion at 2x
        plan.upscale = [{
            "amount": 2,
            "method": "esrgan",
            "strength": 0.2,
            "num_inference_steps": 50,
            "guidance_scale": 10.0,
            "tiling_stride": 256,
        }]
        result = plan.execute(manager)["images"][0]
        result.save(os.path.join(save_dir, "./puppy-plan-xl-upscale-diffusion.png"))
        
        # Upscale again alone
        plan = LayeredInvocation.assemble(
            image=result,
            upscale=[{
                "amount": 2,
                "method": "esrgan",
                "strength": 0.2,
                "tiling_stride": 512,
                "strength": 0.2,
                "num_inference_steps": 50,
                "guidance_scale": 10.0
            }],
            **kwargs
        )
        plan.execute(manager)["images"][0].save(os.path.join(save_dir, "./puppy-plan-xl-upscale-solo.png"))

if __name__ == "__main__":
    main()
