import os
import cv2
import numpy as np

from typing import Iterator, Iterable, Optional, Callable

from datetime import datetime
from PIL import Image
from enfugue.util import logger

__all__ = ["ComputerVision"]

def latent_friendly(number: int) -> int:
    """
    Returns a latent-friendly image size (divisible by 8)
    """
    return (number // 8) * 8

class ComputerVision:
    """
    Provides helper methods for cv2
    """

    @staticmethod
    def show(name: str, image: Image.Image) -> None:
        """
        Shows an image.
        """
        cv2.imshow(name, ComputerVision.convert_image(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def convert_image(image: Image.Image) -> np.ndarray:
        """
        Converts PIL image to OpenCV format.
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def revert_image(array: np.ndarray) -> Image.Image:
        """
        Converts PIL image to OpenCV format.
        """
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    @staticmethod
    def frames_to_video(
        path: str,
        frames: Iterable[Image.Image],
        overwrite: bool = False,
        rate: float = 20.
    ) -> int:
        """
        Saves PIL image frames to an .mp4 video.
        Returns the total size of the video in bytes.
        """
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if os.path.exists(path):
            if not overwrite:
                raise IOError(f"File exists at path {path}, pass overwrite=True to write anyway.")
            os.unlink(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        writer = None
        for frame in frames:
            if writer is None:
                writer = cv2.VideoWriter(path, fourcc, rate, frame.size)
            writer.write(ComputerVision.convert_image(frame))
        if writer is None:
            raise IOError(f"No frames written to {path}")

        writer.release()
        return os.path.getsize(path)

    @staticmethod
    def frames_from_video(
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        resolution: Optional[int] = None,
    ) -> Iterator[Image.Image]:
        """
        Starts a video capture and yields PIL images for each frame.
        """
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
    
    @staticmethod
    def video_to_video(
        source_path: str,
        destination_path: str,
        overwrite: bool = False,
        rate: float = 20.,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        resolution: Optional[int] = None,
        process_frame: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> int:
        """
        Saves PIL image frames to an .mp4 video.
        Returns the total size of the video in bytes.
        """
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
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
