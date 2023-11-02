"""
Uses the invocation planner to test parsing of various layer states from UI
"""
import os

from pibble.util.log import DebugUnifiedLoggingContext

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.util import logger, save_frames_or_image

from PIL import Image

def main() -> None:
    HERE = os.path.dirname(os.path.abspath(__file__))
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(HERE, "test-results", "layers")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = Image.open(os.path.join(HERE, "test-images", "small-inpaint.jpg"))
        mask = Image.open(os.path.join(HERE, "test-images", "small-inpaint-mask-invert.jpg"))
        
        manager = DiffusionPipelineManager()

        def log_invocation(name, invocation):
            logger.info(name)
            processed = invocation.preprocess(manager, raise_when_unused=False)
            formatted = LayeredInvocation.format_serialization_dict(
                save_directory=save_dir,
                save_name=name,
                **processed
            )
            logger.debug(f"{formatted}")
        
        # Layered, image covers entire canvas
        log_invocation(
            "simple",
            LayeredInvocation(
                width=512,
                height=512,
                layers=[
                    {
                        "image": image,
                    }
                ]
            )
        )

        # Layered, image covers part of canvas, no denoising
        log_invocation(
            "outpaint",
            LayeredInvocation(
                width=512,
                height=512,
                layers=[
                    {
                        "image": image,
                        "x": 0,
                        "y": 0,
                        "w": 256,
                        "h": 256
                    }
                ]
            )
        )

        # Layered, image covers part of canvas, denoising
        log_invocation(
            "overpaint",
            LayeredInvocation(
                width=512,
                height=512,
                layers=[
                    {
                        "image": image,
                        "x": 0,
                        "y": 0,
                        "w": 512,
                        "h": 256,
                        "denoise": True
                    }
                ]
            )
        )

        # Layered, image covers part of canvas, joined with mask
        log_invocation(
            "outpaint-merge",
            LayeredInvocation(
                width=512,
                height=512,
                mask={
                    "image": mask,
                    "invert": True
                },
                layers=[
                    {
                        "image": image,
                        "x": 0,
                        "y": 0,
                        "w": 512,
                        "h": 256,
                    }
                ]
            )
        )

        # Layered, image covers part of canvas, joined with mask
        log_invocation(
            "outpaint-merge",
            LayeredInvocation(
                width=512,
                height=512,
                mask={
                    "image": mask,
                    "invert": True
                },
                layers=[
                    {
                        "image": image,
                        "x": 0,
                        "y": 0,
                        "w": 512,
                        "h": 256,
                    }
                ]
            )
        )

        # Layered, image covers entirety of canvas, rembg
        log_invocation(
            "remove_background",
            LayeredInvocation(
                width=512,
                height=512,
                layers=[
                    {
                        "image": image,
                        "remove_background": True
                    }
                ]
            )
        )

        # Layered, two copies of images, merged masks
        log_invocation(
            "remove_background_merge",
            LayeredInvocation(
                width=512,
                height=512,
                mask={
                    "image": mask,
                    "invert": True
                },
                layers=[
                    {
                        "image": image,
                        "x": 128,
                        "w": 512-128,
                        "y": 0,
                        "h": 512
                    },
                    {
                        "image": image,
                        "remove_background": True,
                        "x": 0,
                        "y": 0,
                        "h": 256,
                        "w": 256,
                        "fit": "stretch"
                    }
                ]
            )
        )

if __name__ == "__main__":
    main()
