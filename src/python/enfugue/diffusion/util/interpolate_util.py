from __future__ import annotations

from typing import Optional, Callable, List, Union, Iterable, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "interpolate_frames",
    "reflect_frames"
]

def interpolate_frames(
    frames: List[Image.Image],
    multiplier: Union[List[int], int],
    interpolate: Callable[[Image.Image, Image.Image, int], List[Image]],
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> Iterable[Image.Image]:
    """
    Interpolates an entire video.
    """
    last_frame: Optional[Image.Image] = None
    num_frames = len(frames)

    if isinstance(multiplier, list):
        num_multipliers = len(multiplier)
        if num_multipliers < num_frames - 1:
            multiplier += [multiplier[-1]] * (num_frames - num_multipliers - 1)
        multiplier = multiplier[:num_frames - 1]
        get_multiplier = lambda i: multiplier[i]
        num_interpolations = sum(multiplier)
    else:
        get_multiplier = lambda i: multiplier # type: ignore
        num_interpolations = multiplier * (num_frames - 1)

    interpolation_step = 0

    for i, frame in enumerate(frames):
        if last_frame is not None:
            yield last_frame
            this_multiplier = get_multiplier(i - 1)
            if this_multiplier > 0:
                start = datetime.now()
                interpolated_frames = interpolate(
                    last_frame,
                    frame,
                    this_multiplier
                )
                if progress_callback is not None:
                    duration = (datetime.now() - start).total_seconds()
                    interpolation_step += this_multiplier
                    progress_callback(interpolation_step, num_interpolations, this_multiplier / duration)
                for interpolated_frame in interpolated_frames:
                    yield interpolated_frame
        last_frame = frame
    yield last_frame

def reflect_frames(
    frames: List[Image.Image],
    interpolate: Callable[[Image.Image, Image.Image, int], List[Image]],
    ease_pattern = [2, 2, 1, 1],
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Iterable[Image.Image]:
    """
    Interpolates the start and end of frames, and plays it forward then backwards.
    """
    num_frames = len(frames)
    ease_pattern_length = min(len(ease_pattern), num_frames // 2)
    ease_pattern = ease_pattern[:ease_pattern_length]
    reversed_ease_pattern = list(reversed(ease_pattern))
    multiplier = ease_pattern + [0] * (num_frames - (ease_pattern_length * 2) - 1) + reversed_ease_pattern

    all_frames = []
    for eased_frame in interpolate_frames(
        frames=frames,
        multiplier=multiplier,
        interpolate=interpolate,
        progress_callback=progress_callback
    ):
        yield eased_frame
        all_frames.append(eased_frame)

    all_frames.reverse()
    for frame in all_frames[1:-1]:
        yield frame
