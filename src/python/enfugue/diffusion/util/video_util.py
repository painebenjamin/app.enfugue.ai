from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Iterator, Iterable, List
from datetime import datetime
from enfugue.util import logger

if TYPE_CHECKING:
    from pibble.api.configuration import APIConfiguration
    from PIL.Image import Image

__all__ = [
    "uprate_video",
    "interpolate_frames"
]

def interpolate_frames(
    frames: Iterable[Image],
    multiplier: int = 4,
    configuration: Optional[APIConfiguration] = None
) -> Iterator[Image]:
    """
    Provides a generator for interpolating between multiple frames.
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
        previous_frame = None
        frame_index = 0
        frame_start = datetime.now()
        for frame in frames:
            frame_index += 1
            process_times: List[float] = []
            if previous_frame is not None:
                for i in range(multiplier - 1): # type: ignore[unreachable]
                    process_start = datetime.now()
                    yield process(previous_frame, frame, (i + 1) / multiplier)
                    process_times.append((datetime.now() - process_start).total_seconds())
            yield frame
            frame_time = (datetime.now() - frame_start).total_seconds()
            if previous_frame is not None:
                process_count = len(process_times) # type: ignore[unreachable]
                process_average = sum(process_times) / process_count
                logger.debug(f"Processed frames {frame_index-1}-{frame_index} in {frame_time:.1f} seconds. Interpolated {process_count} frame(s) at a rate of {process_average:.1f} seconds/frame.")
            previous_frame = frame
            frame_start = datetime.now()

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
            frame_index = 0
            frame_start = datetime.now()
            for frame in ComputerVision.frames_from_video(source_path):
                frame_index += 1
                process_times: List[float] = [] 
                if previous_frame is not None:
                    for i in range(target_multiplier - 1): # type: ignore[unreachable]
                        process_start = datetime.now()
                        yield process(previous_frame, frame, (i + 1) / target_multiplier)
                        process_times.append((datetime.now() - process_start).total_seconds())
                yield frame
                frame_time = (datetime.now() - frame_start).total_seconds()
                if previous_frame is not None:
                    process_count = len(process_times) # type: ignore[unreachable]
                    process_average = sum(process_times) / process_count
                    logger.debug(f"Processed frames {frame_index-1}-{frame_index} in {frame_time:.1f} seconds. (interpolated {process_count} frames at a rate of {process_average:.1f} seconds/frame")
                previous_frame = frame
                frame_start = datetime.now()
        
        return ComputerVision.frames_to_video(
            target_path,
            get_frames(),
            overwrite=overwrite,
            rate=source_rate*target_multiplier
        )
