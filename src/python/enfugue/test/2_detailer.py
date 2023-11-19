import os
from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.manager import DiffusionPipelineManager

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "detailer")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        plan = LayeredInvocation.assemble(
            seed=12345,
            width=512,
            height=512,
            prompt="A happy looking puppy",
        )

        plan.execute(
            manager,
            task_callback=lambda arg: logger.info(arg),
        )["images"][0].save(os.path.join(save_dir, "./puppy-plan-upscale.png"))



if __name__ == "__main__":
    main()
