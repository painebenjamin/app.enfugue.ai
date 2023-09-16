from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Iterator

if TYPE_CHECKING:
    from pibble.api.configuration import APIConfiguration
    from PIL.Image import Image

__all__ = [
    "uprate_video"
]

def uprate_video(
    source_path: str,
    source_rate: float,
    target_path: str,
    target_multiplier: int = 4,
    overwrite: bool = False,
    configuration: Optional[APIConfiguration] = None
) -> int:
    """
    Takes a video and increases it's framerate by interpolating intermediate frames.
    """
    import torch
    from enfugue.diffusion.util import get_optimal_device, ComputerVision
    from enfugue.diffusion.support import Interpolator

    device = get_optimal_device()
    if configuration is None:
        from enfugue.util import get_local_configuration
        configuration = get_local_configuration()

    interpolator = Interpolator(
        configuration.get("enfugue.engine.cache", "~/.cache/enfugue/cache"),
        device,
        torch.float16 if device.type == "cuda" else torch.float32
    )
    
    with interpolator.interpolate() as process:
        def get_frames() -> Iterator[Image]:
            previous_frame = None
            for frame in ComputerVision.frames_from_video(source_path):
                if previous_frame is not None:
                    for i in range(target_multiplier - 1): # type: ignore[unreachable]
                        yield process(previous_frame, frame, (i + 1) / target_multiplier)
                yield frame
                previous_frame = frame

        return ComputerVision.frames_to_video(
            target_path,
            get_frames(),
            overwrite=overwrite,
            rate=source_rate*target_multiplier
        )
