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
    @classmethod
    def get_interpolator(cls) -> Interpolator:
        """
        Builds a default interpolator without a configuration passed
        """
        from enfugue.diffusion.support import Interpolator
        return cast(Interpolator, Interpolator.get_default_instance())

    @classmethod
    def interpolate(
        cls,
        frames: Iterable[Image],
        multiplier: Union[int, Tuple[int, ...]] = 2,
        interpolate: Optional[Callable[[Image, Image, float], Image]] = None,
    ) -> Iterator[Image]:
        """
        Provides a generator for interpolating between multiple frames.
        """
        from enfugue.diffusion.util import ComputerVision
        
        if interpolate is None:
            interpolator = cls.get_interpolator()
            with interpolator.interpolate() as process:
                for frame in cls.interpolate(
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
                for frame in cls.interpolate(
                    frames=cls.interpolate(
                        frames=frames,
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
                logger.debug(f"Processed frames {frame_index-1}-{frame_index} in {frame_time:.1f} seconds. Interpolated {process_count} frame(s) at a rate of {process_average:.1f} seconds/frame.")
            previous_frame = frame
            frame_start = datetime.now()

    @classmethod
    def uprate(
        cls,
        source_path: str,
        source_rate: float,
        target_path: str,
        target_multiplier: Union[int, Tuple[int, ...]] = 2,
        overwrite: bool = False,
        encoder: str = "avc1",
    ) -> int:
        """
        Takes a video and increases it's framerate by interpolating intermediate frames.
        """
        multiplier = target_multiplier
        if isinstance(multiplier, tuple):
            from math import prod
            multiplier = prod(multiplier)

        return cls.frames_to_video(
            path=target_path,
            frames=cls.interpolate(
                frames=cls.frames_from_video(source_path),
                multiplier=target_multiplier
            ),
            overwrite=overwrite,
            encoder=encoder,
            rate=source_rate*multiplier
        )

    @classmethod
    def frames_to_video(
        cls,
        path: str,
        frames: Union[str, Iterable[Image.Image]],
        overwrite: bool = False,
        rate: float = 20.0,
        encoder: str = "avc1"
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
        if isinstance(frames, str):
            frames = cls.frames_from_video(frames)
        if ext in [".gif", ".png", ".tiff", ".webp"]:
            frames = [frame for frame in frames]
            frames[0].save(path, loop=0, duration=1000.0/rate, save_all=True, append_images=frames[1:])
            return os.path.getsize(path)
        elif ext != ".mp4":
            raise IOError(f"Unknown file extension {ext}")
        fourcc = cv2.VideoWriter_fourcc(*encoder) # type: ignore
        writer = None
        for frame in frames:
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
    def frames_from_video(
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
    def video_to_video(
        cls,
        source_path: str,
        destination_path: str,
        overwrite: bool = False,
        rate: float = 20.,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        resolution: Optional[int] = None,
        process_frame: Optional[Callable[[Image.Image], Image.Image]] = None,
        encoder: str = "avc1"
    ) -> int:
        """
        Saves PIL image frames to an .mp4 video.
        Returns the total size of the video in bytes.
        """
        import cv2
        from enfugue.diffusion.util import ComputerVision
        if destination_path.startswith("~"):
            destination_path = os.path.expanduser(destination_path)
        if os.path.exists(destination_path):
            if not overwrite:
                raise IOError(f"File exists at destination_path {destination_path}, pass overwrite=True to write anyway.")
            os.unlink(destination_path)

        if source_path.startswith("~"):
            source_path = os.path.expanduser(source_path)
        if not os.path.exists(source_path):
            raise IOError(f"Video at path {source_path} not found or inaccessible")

        frames = 0

        frame_start = 0 if skip_frames is None else skip_frames
        frame_end = None if maximum_frames is None else frame_start + maximum_frames - 1

        frame_string = "end-of-video" if frame_end is None else f"frame {frame_end}"
        logger.debug(f"Reading video file at {source_path} starting from frame {frame_start} until {frame_string}. Will process and write to {destination_path}")

        capture = cv2.VideoCapture(source_path)
        fourcc = cv2.VideoWriter_fourcc(*encoder) # type: ignore
        writer = None

        def process_image(image: Image.Image) -> Image.Image:
            """
            Processes an image frame if requested.
            """
            width, height = image.size
            if resolution is not None:
                ratio = float(resolution) / float(min(width, height))
                height = round(height * ratio)
                width = round(width * ratio)

            image = image.resize((
                latent_friendly(width),
                latent_friendly(height)
            ))

            if process_frame is not None:
                image = process_frame(image)

            return image

        opened = datetime.now()
        started = opened
        last_log = 0
        processed_frames = 0

        while capture.isOpened():
            success, image = capture.read()
            if not success:
                break
            elif frames == 0:
                opened = datetime.now()
                logger.debug("Video opened, iterating through frames.")

            frames += 1
            if frame_start > frames:
                continue
            elif frame_start == frames:
                started = datetime.now()
                logger.debug(f"Beginning processing from frame {frames}")

            image = process_image(ComputerVision.revert_image(image))
            processed_frames += 1
            
            if writer is None:
                writer = cv2.VideoWriter(destination_path, fourcc, rate, image.size)
            
            writer.write(ComputerVision.convert_image(image))
            
            if last_log < processed_frames - rate:
                unit = "frames/sec"
                process_rate = processed_frames / (datetime.now() - started).total_seconds()
                if process_rate < 1.0:
                    unit = "sec/frame"
                    process_rate = 1.0 / process_rate

                logger.debug(f"Processed {processed_frames} at {process_rate:.2f} {unit}")
                last_log = processed_frames
            
            if frame_end is not None and frames >= frame_end:
                break
        if writer is None:
            raise IOError(f"No frames written to path {destination_path}")

        writer.release()
        capture.release()

        if frames == 0:
            raise IOError(f"No frames were read from video at {source_path}")

        return os.path.getsize(destination_path)
