import io
import os
import PIL
import requests

from enfugue.diffusion.manager import DiffusionPipelineManager

from pibble.util.log import DebugUnifiedLoggingContext

CAPTIONS = [
    "a cat in a hat",
    "a pretty pony",
    "a nice sunset",
    "a pretty lady",
    "a puppy"
]


def main() -> None:
    with DebugUnifiedLoggingContext():
        manager = DiffusionPipelineManager()
        with manager.caption_upsampler.upsampler() as upsample:
            for caption in CAPTIONS:
                for i in range(NUM_CAPTIONS_PER_PROMPT):
                    upsampled = upsample(caption)
                    print(f"{caption} = {upsampled}")


if __name__ == "__main__":
    main()
