"""
Uses the engine to exercise layered plan instantiation and execution
"""
import os
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.manager import DiffusionPipelineManager

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        manager = DiffusionPipelineManager()

        # Re-diffused upscaled image
        plan = LayeredInvocation.assemble(
            seed=12345,
            width=512,
            height=512,
            prompt="A happy looking puppy",
            upscale={
                "method": "esrgan",
                "amount": 2,
                "strength": 0.2
            }
        )

        plan.execute(
            manager,
            task_callback=lambda arg: logger.info(arg),
        )["images"][0].save(os.path.join(save_dir, "./puppy-plan-upscale.png"))



if __name__ == "__main__":
    main()
