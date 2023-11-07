from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional, Iterator, Callable, Iterable
from enfugue.util import logger

if TYPE_CHECKING:
    from PIL.Image import Image
    import cv2

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

    def save(
        self,
        path: str,
        overwrite: bool = False,
        rate: float = 20.0,
        encoder: str = "avc1",
    ) -> int:
        """
        Saves PIL image frames to a video.
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
        on_open: Optional[Callable[[cv2.VideoCapture], None]] = None,
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
        frame_end = None if maximum_frames is None else frame_start + maximum_frames - 1

        frame_string = "end-of-video" if frame_end is None else f"frame {frame_end}"
        logger.debug(f"Reading video file at {path} starting from frame {frame_start} until {frame_string}")

        capture = cv2.VideoCapture(path)
        if on_open is not None:
            on_open(capture)

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
        on_open: Optional[Callable[[cv2.VideoCapture], None]] = None,
    ) -> Video:
        """
        Uses Video.frames_from_file and instantiates a Video object.
        """
        return cls(
            frames=cls.file_to_frames(
                path=path,
                skip_frames=skip_frames,
                maximum_frames=maximum_frames,
                resolution=resolution,
                on_open=on_open,
            )
        )
