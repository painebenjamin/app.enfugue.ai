import io
import os
import PIL
import torch
import requests

from enfugue.util import logger
from einops import rearrange
from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test-results", "optical-flow"
        )
        image_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test-images"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from enfugue.diffusion.util import Video
        video = Video.from_file(os.path.join(image_dir, "pexels-dolphin.mp4"))
        logger.info(f"Testing sparse")
        Video(video.sparse_flow_image()).save(
            os.path.join(save_dir, "sparse.gif"),
            rate=16.0,
            overwrite=True
        )
        for method in ["lucas-kanade", "farneback", "rlof"]:
            logger.info(f"Testing method {method}")
            Video(video.dense_flow_image(method=method)).save(
                os.path.join(save_dir, f"{method}.gif"),
                rate=16.0,
                overwrite=True
            )

if __name__ == "__main__":
    main()
