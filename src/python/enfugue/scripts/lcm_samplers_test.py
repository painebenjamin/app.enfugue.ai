from typing import get_args

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import GridMaker, Video
from enfugue.diffusion.constants import SCHEDULER_LITERAL

from pibble.util.log import DebugUnifiedLoggingContext

manager = DiffusionPipelineManager()
manager.animator = "epicrealism_pureEvolutionV5.safetensors"
manager.animator_vae = "ema"
manager.motion_module = "temporaldiffMotion_v10.ckpt"
manager.lora = [
    ("epi_noiseoffset2.safetensors", 0.99),
    ("lcm-lora-sdv1-5.safetensors", 1.0)
]
manager.inversion = ["BadDream.pt", "UnrealisticDream.pt"]

grid = GridMaker(
    use_video=True,
    prompt="a goldfish swimming in a fishbowl",
    negative_prompt="BadDream, UnrealisticDream",
    animation_frames=24,
    frame_window_size=16,
    frame_window_stride=4,
    width=512,
    height=512,
    num_inference_steps=5,
    guidance_scale=1.25,
    seed=12345,
    grid_size=512,
    grid_columns=6,
    font_size=32,
)

# Execute once so timing is even
grid.execute(manager, {})

with DebugUnifiedLoggingContext():
    video_frames = grid.execute(
        manager,
        *[
            {
                "scheduler": scheduler,
                "label": scheduler.upper()
            }
            for scheduler in get_args(SCHEDULER_LITERAL)
            if scheduler not in [
               "dpmsde", "adpmd", "adpmdk", "pndm"
            ] # skip bad ones
        ]
    )

Video(video_frames).save(
    "./grid.mp4",
    overwrite=True,
    rate=8.0
)
