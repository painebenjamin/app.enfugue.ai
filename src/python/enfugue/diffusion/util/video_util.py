from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional, Iterator, Iterable, List, Callable, Union, Tuple, cast
from datetime import datetime
from enfugue.util import logger

if TYPE_CHECKING:
    from enfugue.diffusion.support import Interpolator
    from PIL.Image import Image

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
    def __init__(self, frames: Iterable[Image]) -> None:
        self.frames = frames

    @classmethod
    def get_interpolator(cls) -> Interpolator:
        """
        Builds a default interpolator without a configuration passed
        """
        from enfugue.diffusion.support import Interpolator
        return cast(Interpolator, Interpolator.get_default_instance())

    def interpolate(
        self,
        frames: Optional[Iterable[Image]] = None,
        multiplier: Union[int, Tuple[int, ...]] = 2,
        interpolate: Optional[Callable[[Image, Image, float], Image]] = None,
    ) -> Iterator[Image]:
        """
        Provides a generator for interpolating between multiple frames.
        """
        from enfugue.diffusion.util import ComputerVision

        if frames is None:
            for frame in self.interpolate(
                frames=self.frames, # type: ignore[arg-type]
                multiplier=multiplier,
                interpolate=interpolate
            ):
                yield frame
            return

        if interpolate is None:
            interpolator = self.get_interpolator()
            with interpolator.interpolate() as process:
                for frame in self.interpolate(
                    frames=frames,
                    multiplier=multiplier,
                    interpolate=process
                ):
                    yield frame
                return

        if isinstance(multiplier, tuple):
            if len(multiplier) == 1:
                multiplier = multiplier[0]
            else:
                this_multiplier = multiplier[0]
                recursed_multiplier = multiplier[1:]
                for frame in self.interpolate(
                    frames=self.interpolate(
                        frames=frames, # type: ignore[arg-type]
                        multiplier=recursed_multiplier,
                        interpolate=interpolate
                    ),
                    multiplier=this_multiplier,
                    interpolate=interpolate
                ):
                    yield frame
                return

        previous_frame = None
        frame_index = 0
        frame_start = datetime.now()
        for frame in frames:
            frame_index += 1
            process_times: List[float] = []
            if previous_frame is not None:
                for i in range(multiplier - 1): # type: ignore[unreachable]
                    process_start = datetime.now()
                    yield interpolate(previous_frame, frame, (i + 1) / multiplier)
                    process_times.append((datetime.now() - process_start).total_seconds())
            yield frame
            frame_time = (datetime.now() - frame_start).total_seconds()
            if previous_frame is not None:
                process_count = len(process_times) # type: ignore[unreachable]
                process_average = sum(process_times) / process_count
                logger.debug(
                    f"Processed frames {frame_index-1}-{frame_index} in {frame_time:.1f} seconds. " +
                    f"Interpolated {process_count} frame(s) at a rate of {process_average:.1f} seconds/frame."
                )
            previous_frame = frame
            frame_start = datetime.now()

    def loop(
        self,
        ease_frames: int = 2,
        double_ease_frames: int = 1,
        hold_frames: int = 0,
        interpolate: Optional[Callable[[Image, Image, float], Image]] = None,
    ) -> Iterable[Image.Image]:
        """
        Takes a video and creates a gently-looping version of it.
        """
        if interpolate is None:
            interpolator = self.get_interpolator()
            with interpolator.interpolate() as process:
                for frame in self.loop(
                    ease_frames=ease_frames,
                    double_ease_frames=double_ease_frames,
                    hold_frames=hold_frames,
                    interpolate=process
                ):
                    yield frame
                return

        # Memoized frames
        frame_list: List[Image.Image] = [frame for frame in self.frames]
        
        if double_ease_frames:
            double_ease_start_frames, frame_list = frame_list[:double_ease_frames], frame_list[double_ease_frames:]
        else:
            double_ease_start_frames = []
        if ease_frames:
            ease_start_frames, frame_list = frame_list[:ease_frames], frame_list[ease_frames:]
        else:
            ease_start_frames = []

        if double_ease_frames:
            frame_list, double_ease_end_frames = frame_list[:-double_ease_frames], frame_list[-double_ease_frames:]
        else:
            double_ease_end_frames = []
        if ease_frames:
            frame_list, ease_end_frames = frame_list[:-ease_frames], frame_list[-ease_frames:]
        else:
            ease_end_frames = []

        # Interpolate frames
        double_ease_start_frames = [
            frame for frame in self.interpolate(
                frames=double_ease_start_frames,
                multiplier=(2,2),
                interpolate=interpolate
            )
        ]
        ease_start_frames = [
            frame for frame in self.interpolate(
                frames=ease_start_frames,
                multiplier=2,
                interpolate=interpolate
            )
        ]
        ease_end_frames = [
            frame for frame in self.interpolate(
                frames=ease_end_frames,
                multiplier=2,
                interpolate=interpolate
            )
        ]
        double_ease_end_frames = [
            frame for frame in self.interpolate(
                frames=double_ease_end_frames,
                multiplier=(2,2),
                interpolate=interpolate
            )
        ]

        # Return to one list
        frame_list = double_ease_start_frames + ease_start_frames + frame_list + ease_end_frames + double_ease_end_frames

        # Iterate
        for frame in frame_list:
            yield frame

        # Hold on final frame
        for i in range(hold_frames):
            yield frame_list[-1]

        # Reverse the frames
        frame_list.reverse()
        for frame in frame_list[1:-1]:
            yield frame

        # Hold on first frame
        for i in range(hold_frames):
            yield frame_list[-1]

    def save(
        self,
        path: str,
        overwrite: bool = False,
        rate: float = 20.0,
        encoder: str = "avc1",
    ) -> int:
        """
        Saves PIL image frames to an .mp4 video.
        Returns the total size of the video in bytes.
        """
        import cv2
        from enfugue.diffusion.util import ComputerVision
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if os.path.exists(path):
            if not overwrite:
                raise IOError(f"File exists at path {path}, pass overwrite=True to write anyway.")
            os.unlink(path)
        basename, ext = os.path.splitext(os.path.basename(path))
        if ext in [".gif", ".png", ".tiff", ".webp"]:
            frames = [frame for frame in self.frames]
            frames[0].save(path, loop=0, duration=1000.0/rate, save_all=True, append_images=frames[1:])
            return os.path.getsize(path)
        elif ext != ".mp4":
            raise IOError(f"Unknown file extension {ext}")
        fourcc = cv2.VideoWriter_fourcc(*encoder) # type: ignore
        writer = None

        for frame in self.frames:
            if writer is None:
                writer = cv2.VideoWriter(path, fourcc, rate, frame.size) # type: ignore[union-attr]
            writer.write(ComputerVision.convert_image(frame))

        if writer is None:
            raise IOError(f"No frames written to {path}")

        writer.release()

        if not os.path.exists(path):
            raise IOError(f"Nothing was written to {path}")
        return os.path.getsize(path)

    @classmethod
    def file_to_frames(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        resolution: Optional[int] = None,
    ) -> Iterator[Image.Image]:
        """
        Starts a video capture and yields PIL images for each frame.
        """
        import cv2
        from enfugue.diffusion.util import ComputerVision

        if path.startswith("~"):
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise IOError(f"Video at path {path} not found or inaccessible")
        
        basename, ext = os.path.splitext(os.path.basename(path))
        if ext in [".gif", ".png", ".tiff", ".webp"]:
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
        frame_end = None if maximum_frames is None else frame_start + maximum_frames - 1

        frame_string = "end-of-video" if frame_end is None else f"frame {frame_end}"
        logger.debug(f"Reading video file at {path} starting from frame {frame_start} until {frame_string}")

        capture = cv2.VideoCapture(path)

        def resize_image(image: Image.Image) -> Image.Image:
            """
            Resizes an image frame if requested.
            """
            if resolution is None:
                return image

            width, height = image.size
            ratio = float(resolution) / float(min(width, height))
            height = round(height * ratio)
            width = round(width * ratio)
            return image.resize((width, height))

        while capture.isOpened():
            success, image = capture.read()
            if not success:
                break
            elif frames == 0:
                logger.debug("First frame captured, iterating.")

            frames += 1
            if frame_start > frames:
                continue

            yield resize_image(ComputerVision.revert_image(image))

            if frame_end is not None and frames >= frame_end:
                break

        capture.release()
        if frames == 0:
            raise IOError(f"No frames were read from video at {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        resolution: Optional[int] = None,
    ) -> Video:
        """
        Uses Video.frames_from_file and instantiates a Video object.
        """
        return cls(
            frames=cls.file_to_frames(
                path=path,
                skip_frames=skip_frames,
                maximum_frames=maximum_frames
            )
        )
