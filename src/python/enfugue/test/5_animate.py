"""
Tests automatic loading of motion module/animator pipeline
"""
import os
import PIL

from enfugue.util import logger
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.diffusion.constants import *
from enfugue.diffusion.util import Video

from pibble.util.log import DebugUnifiedLoggingContext
from pibble.util.numeric import human_size

PROMPT = "a beautiful woman smiling, open mouth, bright teeth"
FRAMES = 16
RATE = 8.0

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "test-results", "animation")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with DebugUnifiedLoggingContext():
        with DiffusionEngine.debug() as engine:
            base_result = engine(
                animation_frames=FRAMES,
                seed=123456,
                prompt=PROMPT,
            )["images"]

            logger.debug(f"Result is {len(base_result)} frames, writing.")

            target = os.path.join(output_dir, "base.mp4")
            size = Video(base_result).save(
                target,
                rate=RATE,
                overwrite=True,
            )
            print(f"Wrote {FRAMES} frames to {target} ({human_size(size)}")

if __name__ == "__main__":
    main()
