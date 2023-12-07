import io
import os
import PIL
import requests

from enfugue.diffusion.manager import DiffusionPipelineManager

from pibble.util.log import DebugUnifiedLoggingContext

CAPTIONS = [
    "a golden retriever",
    "a cat in a hat",
    "an elderly couple walking through a park"
]

NUM_CAPTIONS_PER_PROMPT = 4

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
