import os
import math
import torch

from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.util import logger
from enfugue.diffusion.util import Chunker

def main() -> None:
    with DebugUnifiedLoggingContext():
        chunker = Chunker(
            width=512,
            height=512,
            frames=32,
            size=512,
            stride=64,
            frame_size=16,
            frame_stride=4,
            loop=True,
        )
        for chunk in chunker.get_chunks(0):
            logger.info(f"{chunk}")

if __name__ == "__main__":
    main()
