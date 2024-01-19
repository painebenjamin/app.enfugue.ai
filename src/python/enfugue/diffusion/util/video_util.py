from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Iterator, Callable, Iterable, Literal, List, Union, Tuple
from enfugue.util import logger, reiterator
from pibble.util.helpers import OutputCatcher

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
    from PIL.Image import Image
    from moviepy.editor import VideoFileClip
    from enfugue.util.images import IMAGE_FIT_LITERAL, IMAGE_ANCHOR_LITERAL
    from enfugue.diffusion.util.audio_util.helper import Audio

__all__ = ["Video"]

def latent_friendly(number: int) -> int:
    """
    Returns a latent-friendly image size (divisible by 8)
    """
    return (number // 8) * 8

class Video:
    """
    Provides helper methods for video
    """
    audio: Optional[Audio] = None

    def __init__(
        self,
        frames: Iterable[Image],
        frame_rate: Optional[float]=None,
        audio: Optional[Audio]=None,
        audio_frames: Optional[Iterable[Tuple[float]]]=None,
        audio_rate: Optional[int]=None,
    ) -> None:
        self.frames = reiterator(frames)
        self.frame_rate = frame_rate
        if audio is not None:
            self.audio = audio
        elif audio_frames is not None:
            from enfugue.diffusion.util.audio_util import Audio
            self.audio = Audio(frames=audio_frames, rate=audio_rate)
        else:
            self.audio = None

    def save(
        self,
        path: str,
        overwrite: bool = False,
        rate: float=20.0,
        encoder: str="avc1",
    ) -> int:
        """
        Saves PIL image frames to a video.
        Returns the total size of the video in bytes.
        """
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if os.path.exists(path):
            if not overwrite:
                raise IOError(f"File exists at path {path}, pass overwrite=True to write anyway.")
            os.unlink(path)
        basename, ext = os.path.splitext(os.path.basename(path))
        if ext in [".gif", ".png", ".tiff", ".webp"]:
            frames = [frame for frame in self.frames] # type: ignore[attr-defined]
            if rate > 50:
                logger.warning(f"Rate {rate} exceeds maximum frame rate (50), clamping.")
                rate = 50
            frames[0].save(path, loop=0, duration=1000.0/rate, save_all=True, append_images=frames[1:])
            return os.path.getsize(path)
        elif ext not in [".mp4", ".ogg", ".webm"]:
            raise IOError(f"Unknown file extension {ext}")

        from moviepy.editor import ImageSequenceClip
        import numpy as np

        clip_frames = [np.array(frame) for frame in self.frames] # type: ignore[attr-defined]

        catcher = OutputCatcher()
        with catcher:
            try:
                clip = ImageSequenceClip(clip_frames, fps=rate)

                if self.audio is not None:
                    clip.audio = self.audio.get_composite_clip(
                        maximum_seconds=len(clip_frames)/rate
                    )

                clip.write_videofile(path, fps=rate, logger=None)

                if not os.path.exists(path):
                    raise IOError(f"Nothing was written to {path}")
                return os.path.getsize(path)
            finally:
                out, err = catcher.output()
                if out:
                    logger.debug(f"stdout: {out}")
                if err:
                    logger.info(f"stderr (may not be an error:) {err}")
                catcher.clean()

    @classmethod
    def file_to_frames(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        divide_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fit: Optional[IMAGE_FIT_LITERAL] = None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        on_open: Optional[Callable[[VideoFileClip], None]] = None,
    ) -> Iterator[Image]:
        """
        Starts a video capture and yields PIL images for each frame.
        """
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise IOError(f"Video at path {path} not found or inaccessible")
        
        basename, ext = os.path.splitext(os.path.basename(path))
        if ext in [".gif", ".png", ".apng", ".tiff", ".webp", ".avif"]:
            from PIL import Image
            image = Image.open(path)
            for i in range(image.n_frames):
                image.seek(i)
                copied = image.copy()
                copied = copied.convert("RGBA")
                yield copied
            return

        frames = 0

        frame_start = 0 if skip_frames is None else skip_frames
        frame_end = None if maximum_frames is None else frame_start + (maximum_frames * (1 if not divide_frames else divide_frames)) - 1

        frame_string = "end-of-video" if frame_end is None else f"frame {frame_end}"
        logger.debug(f"Reading video file at {path} starting from frame {frame_start} until {frame_string}")

        from moviepy.editor import VideoFileClip
        from PIL import Image

        clip = VideoFileClip(path)
        if on_open is not None:
            on_open(clip)

        def resize_image(image: Image.Image) -> Image.Image:
            """
            Resizes an image frame if requested.
            """
            if width is None or height is None:
                return image

            from enfugue.util import fit_image
            return fit_image(
                image,
                width=width,
                height=height,
                fit=fit,
                anchor=anchor
            )

        for frame in clip.iter_frames():
            if frames == 0:
                logger.debug("First frame captured, iterating.")

            frames += 1
            if frame_start > frames:
                continue
            if divide_frames is not None and (frames - frame_start) % divide_frames != 0:
                continue
            yield resize_image(Image.fromarray(frame))

            if frame_end is not None and frames >= frame_end:
                break

        if frames == 0:
            raise IOError(f"No frames were read from video at {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        divide_frames: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fit: Optional[IMAGE_FIT_LITERAL] = None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        on_open: Optional[Callable[[VideoFileClip], None]] = None,
    ) -> Video:
        """
        Uses Video.frames_from_file and instantiates a Video object.
        """
        return cls(
            frames=cls.file_to_frames(
                path=path,
                skip_frames=skip_frames,
                divide_frames=divide_frames,
                maximum_frames=maximum_frames,
                width=width,
                height=height,
                fit=fit,
                anchor=anchor,
                on_open=on_open,
            )
        )

    def dense_flow(
        self,
        method: Literal["dense-lucas-kanade", "farneback", "rlof"] = "dense-lucas-kanade",
        farneback_params: List[Union[int, float]] = [0.5, 3, 15, 3, 5, 1.2, 0],
    ) -> Iterator[NDArray]:
        """
        Calculates dense flow between frames and yields the numpy array
        """
        from enfugue.diffusion.util.vision_util import ComputerVision
        from PIL.Image import Image
        last_frame: Optional[Image] = None
        for frame in self.frames: # type: ignore[attr-defined]
            if last_frame is not None:
                flow = ComputerVision.dense_flow(
                    image_1=last_frame,
                    image_2=frame,
                    method=method,
                    farneback_params=farneback_params
                )
                yield flow
            last_frame = frame

    def sparse_flow(
        self,
        feature_max_corners: int=100,
        feature_quality_level: float=0.3,
        feature_min_distance: int=7,
        feature_block_size: int=10,
        lk_window_size: Tuple[int, int]=(15, 15),
        lk_max_level: int=2,
        lk_criteria: Tuple[int, int, float]=(3, 10, 0.03),
    ) -> Iterator[NDArray]:
        """
        Calculates dense flow between frames and yields the numpy array
        """
        from enfugue.diffusion.util.vision_util import ComputerVision
        from PIL.Image import Image
        last_frame: Optional[Image] = None
        features: Optional[NDArray] = None
        for frame in self.frames: # type: ignore[attr-defined]
            if last_frame is not None:
                flow, features = ComputerVision.sparse_flow(
                    image_1=last_frame,
                    image_2=frame,
                    features=features, # type: ignore[arg-type]
                    lk_window_size=lk_window_size,
                    lk_max_level=lk_max_level,
                    lk_criteria=lk_criteria,
                    feature_max_corners=feature_max_corners,
                    feature_quality_level=feature_quality_level,
                    feature_min_distance=feature_min_distance,
                    feature_block_size=feature_block_size
                )
                yield flow
            last_frame = frame

    def dense_flow_image(
        self,
        method: Literal["dense-lucas-kanade", "farneback", "rlof"] = "dense-lucas-kanade",
        farneback_params: List[Union[int, float]] = [0.5, 3, 15, 3, 5, 1.2, 0],
    ) -> Iterator[Image]:
        """
        Calculates dense flow and returns as images
        """
        from enfugue.diffusion.util.vision_util import ComputerVision
        for flow in self.dense_flow(method=method, farneback_params=farneback_params):
            yield ComputerVision.flow_to_image(flow)

    def sparse_flow_image(
        self,
        feature_max_corners: int=100,
        feature_quality_level: float=0.3,
        feature_min_distance: int=7,
        feature_block_size: int=7,
        lk_window_size: Tuple[int, int]=(15, 15),
        lk_max_level: int=2,
        lk_criteria: Tuple[int, int, float]=(3, 10, 0.03),
    ) -> Iterator[NDArray]:
        """
        Calculates sparse flow and returns as images
        """
        from enfugue.diffusion.util.vision_util import ComputerVision
        for flow in self.sparse_flow(
            feature_max_corners=feature_max_corners,
            feature_quality_level=feature_quality_level,
            feature_min_distance=feature_min_distance,
            feature_block_size=feature_block_size,
            lk_window_size=lk_window_size,
            lk_max_level=lk_max_level,
            lk_criteria=lk_criteria
        ):
            yield ComputerVision.flow_to_image(flow)
