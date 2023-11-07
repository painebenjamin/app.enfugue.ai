from __future__ import annotations

import io
import os
import math

from typing import Optional, Literal, Union, List, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

from pibble.resources.retriever import Retriever
from pibble.util.strings import get_uuid

__all__ = [
    "fit_image",
    "feather_mask",
    "tile_image",
    "image_from_uri",
    "images_are_equal",
    "get_frames_or_image",
    "get_frames_or_image_from_file",
    "save_frames_or_image",
    "create_mask",
    "scale_image",
    "get_image_metadata",
    "redact_images_from_metadata",
    "IMAGE_FIT_LITERAL",
    "IMAGE_ANCHOR_LITERAL",
]

IMAGE_FIT_LITERAL = Literal["actual", "stretch", "cover", "contain"]
IMAGE_ANCHOR_LITERAL = Literal[
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center-center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

def fit_image(
    image: Union[Image, List[Image]],
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
    offset_left: Optional[int] = None,
    offset_top: Optional[int] = None
) -> Image:
    """
    Given an image of unknown size, make it a known size with optional fit parameters.
    """
    if not isinstance(image, list):
        if getattr(image, "n_frames", 1) > 1:
            frames = []
            for i in range(image.n_frames):
                image.seek(i)
                frames.append(image.copy().convert("RGBA"))
            image = frames
    if isinstance(image, list):
        return [
            fit_image(
                img,
                width=width,
                height=height,
                fit=fit,
                anchor=anchor,
                offset_left=offset_left,
                offset_top=offset_top,
            )
            for img in image
        ]

    from PIL import Image

    if fit is None or fit == "actual":
        left, top = 0, 0
        crop_left, crop_top = 0, 0
        image_width, image_height = image.size

        if anchor is not None:
            top_part, left_part = anchor.split("-")

            if top_part == "center":
                top = height // 2 - image_height // 2
            elif top_part == "bottom":
                top = height - image_height

            if left_part == "center":
                left = width // 2 - image_width // 2
            elif left_part == "right":
                left = width - image_width

        blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left
        if image.mode == "RGBA":
            blank_image.paste(image, (left, top), image)
        else:
            blank_image.paste(image, (left, top))

        return blank_image

    elif fit == "contain":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width = int(image_width * width_ratio)
        horizontal_image_height = int(image_height * width_ratio)
        vertical_image_width = int(image_width * height_ratio) 
        vertical_image_height = int(image_height * height_ratio)
        top, left = 0, 0
        direction = None
        if width >= horizontal_image_width and height >= horizontal_image_height:
            input_image = image.resize((horizontal_image_width, horizontal_image_height))
            if anchor is not None:
                top_part, _ = anchor.split("-")
                if top_part == "center":
                    top = height // 2 - horizontal_image_height // 2
                elif top_part == "bottom":
                    top = height - horizontal_image_height
        elif width >= vertical_image_width and height >= vertical_image_height:
            input_image = image.resize((vertical_image_width, vertical_image_height))
            if anchor is not None:
                _, left_part = anchor.split("-")
                if left_part == "center":
                    left = width // 2 - vertical_image_width // 2
                elif left_part == "right":
                    left = width - vertical_image_width

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left

        blank_image = Image.new("RGBA", (width, height))
        if input_image.mode == "RGBA":
            blank_image.paste(input_image, (left, top), input_image)
        else:
            blank_image.paste(input_image, (left, top))

        return blank_image

    elif fit == "cover":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width = math.ceil(image_width * width_ratio)
        horizontal_image_height = math.ceil(image_height * width_ratio)
        vertical_image_width = math.ceil(image_width * height_ratio)
        vertical_image_height = math.ceil(image_height * height_ratio)
        top, left = 0, 0
        direction = None
        if width <= horizontal_image_width and height <= horizontal_image_height:
            input_image = image.resize((horizontal_image_width, horizontal_image_height))
            if anchor is not None:
                top_part, _ = anchor.split("-")
                if top_part == "center":
                    top = height // 2 - horizontal_image_height // 2
                elif top_part == "bottom":
                    top = height - horizontal_image_height
        elif width <= vertical_image_width and height <= vertical_image_height:
            input_image = image.resize((vertical_image_width, vertical_image_height))
            if anchor is not None:
                _, left_part = anchor.split("-")
                if left_part == "center":
                    left = width // 2 - vertical_image_width // 2
                elif left_part == "right":
                    left = width - vertical_image_width
        else:
            input_image = image.resize((width, height))  # We're probably off by a pixel

        if offset_top is not None:
            top += offset_top
        if offset_left is not None:
            left += offset_left

        blank_image = Image.new("RGBA", (width, height))
        if input_image.mode == "RGBA":
            blank_image.paste(input_image, (left, top), input_image)
        else:
            blank_image.paste(input_image, (left, top))

        return blank_image

    elif fit == "stretch":
        return image.resize((width, height)).convert("RGBA")

    else:
        raise ValueError(f"Unknown fit {fit}")

def feather_mask(
    image: Union[Image, List[Image]]
) -> Union[Image, List[Image]]:
    """
    Given an image, create a feathered binarized mask by 'growing' the black/white pixel sections.
    """
    if not isinstance(image, list):
        if getattr(image, "n_frames", 1) > 1:
            frames = []
            for i in range(image.n_frames):
                image.seek(i)
                frames.append(image.copy().convert("RGBA"))
            image = frames
    if isinstance(image, list):
        return [
            feather_mask(img)
            for img in image
        ]

    width, height = image.size

    mask = image.convert("L")
    feathered = mask.copy()

    for x in range(width):
        for y in range(height):
            if mask.getpixel((x, y)) == (0):
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and mask.getpixel((nx, ny)) == 255:
                        feathered.putpixel((x, y), (255))
                        break
    return feathered

def tile_image(image: Image, tiles: Union[int, Tuple[int, int]]) -> Image:
    """
    Given an image and number of tiles, create a tiled image.
    Accepts either an integer (squre tiles) or tuple (rectangular)
    """
    from PIL import Image
    width, height = image.size
    if isinstance(tiles, tuple):
        width_tiles, height_tiles = tiles
    else:
        width_tiles, height_tiles = tiles, tiles
    tiled = Image.new(image.mode, (width * width_tiles, height * height_tiles))
    for i in range(width_tiles):
        for j in range(height_tiles):
            tiled.paste(image, (i * width, j * height))
    return tiled

def image_from_uri(uri: str) -> Image:
    """
    Loads an image using the pibble reteiever; works with http, file, ftp, ftps, sftp, and s3
    """
    from PIL import Image
    return Image.open(io.BytesIO(Retriever.get(uri).all()))

def images_are_equal(image_1: Image, image_2: Image) -> bool:
    """
    Determines if two images are equal.
    """
    from PIL import ImageChops
    if image_1.height != image_2.height or image_1.width != image_2.width:
        return False
    if image_1.mode == image_2.mode == "RGBA":
        image_1_alpha = [p[3] for p in image_1.getdata()]
        image_2_alpha = [p[3] for p in image_2.getdata()]
        if image_1_alpha != image_2_alpha:
            return False
    return not ImageChops.difference(
        image_1.convert("RGB"), image_2.convert("RGB")
    ).getbbox()

def image_pixelize(image: Image, factor: int = 2, exact: bool = True) -> None:
    """
    Makes an image pixelized by downsizing and upsizing by a factor.
    """
    from PIL import Image
    from PIL.Image import Resampling
    width, height = image.size
    downsample_width = width // 2 ** factor
    downsample_height = height // 2 ** factor
    upsample_width = downsample_width * 2 ** factor if exact else width
    upsample_height = downsample_height * 2 ** factor if exact else height
    image = image.resize((downsample_width, downsample_height), resample=Resampling.NEAREST)
    image = image.resize((upsample_width, upsample_height), resample=Resampling.NEAREST)
    return image

def get_frames_or_image(image: Union[Image, List[Image]]) -> Union[Image, List[Image]]:
    """
    Makes sure an image is a list of images if it has more than one frame
    """
    if not isinstance(image, list):
        if getattr(image, "n_frames", 1) > 1:
            def get_frame(i: int) -> Image:
                image.seek(i) # type: ignore[union-attr]
                return image.copy().convert("RGB") # type: ignore[union-attr]
            return [
                get_frame(i)
                for i in range(image.n_frames)
            ]
    return image

def save_frames_or_image(
    image: Union[Image, List[Image]],
    directory: str,
    name: Optional[str]=None,
    video_format: str="webp",
    image_format: str="png"
) -> str:
    """
    Saves frames to image or video 
    """
    image = get_frames_or_image(image)
    if name is None:
        name = get_uuid()
    if isinstance(image, list):
        from enfugue.diffusion.util.video_util import Video
        path = os.path.join(directory, f"{name}.{video_format}")
        Video(image).save(path)
    else:
        path = os.path.join(directory, f"{name}.{image_format}")
        image.save(path)
    return path

def get_frames_or_image_from_file(path: str) -> Union[Image, List[Image]]:
    """
    Opens a file to a single image or multiple
    """
    if path.startswith("data:"):
        # Should be a video
        if not path.startswith("data:video"):
            raise IOError(f"Received non-video data in video handler: {path}")
        # Dump to tempfile
        from tempfile import mktemp
        from base64 import b64decode
        header, _, data = path.partition(",")
        fmt, _, encoding = header.partition(";")
        _, _, file_ext = fmt.partition("/")
        dump_file = mktemp(f".{file_ext}")
        try:
            with open(dump_file, "wb") as fh:
                fh.write(b64decode(data))
            from enfugue.diffusion.util.video_util import Video
            return list(Video.file_to_frames(dump_file))
        finally:
            os.unlink(dump_file)
    else:
        name, ext = os.path.splitext(path)
            
        if ext in [".webp", ".webm", ".mp4", ".avi", ".mov", ".gif", ".m4v", ".mkv", ".ogg"]:
            from enfugue.diffusion.util.video_util import Video
            return list(Video.file_to_frames(path))
        else:
            from PIL import Image
            return Image.open(path)

def create_mask(
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int
) -> Image:
    """
    Creates a mask from 6 dimensions
    """
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(left, top), (right, bottom)], fill="#ffffff")
    return image

def scale_image(image: Image, scale: Union[int, float]) -> Image:
    """
    Scales an image proportionally.
    """
    width, height = image.size
    scaled_width = 8 * round((width * scale) / 8)
    scaled_height = 8 * round((height * scale) / 8)
    return image.resize((scaled_width, scaled_height))

def get_image_metadata(image: Union[str, Image, List[Image]]) -> Dict[str, Any]:
    """
    Gets metadata from an image
    """
    if isinstance(image, str):
        return get_image_metadata(get_frames_or_image_from_file(image))
    elif isinstance(image, list):
        (width, height) = image[0].size
        return {
            "width": width,
            "height": height,
            "frames": len(image),
            "metadata": getattr(image[0], "text", {}),
        }
    else:
        (width, height) = image.size
        return {
            "width": width,
            "height": height,
            "metadata": getattr(image, "text", {})
        }

def redact_images_from_metadata(metadata: Dict[str, Any]) -> None:
    """
    Removes images from a metadata dictionary
    """
    for key in ["image", "mask"]:
        image = metadata.get(key, None)
        if image is not None:
            if isinstance(image, dict):
                image["image"] = get_image_metadata(image["image"])
            elif isinstance(image, str):
                metadata[key] = get_image_metadata(metadata[key])
            else:
                metadata[key] = get_image_metadata(metadata[key])
    if "control_images" in metadata:
        for i, control_dict in enumerate(metadata["control_images"]):
            control_dict["image"] = get_image_metadata(control_dict["image"])
    if "ip_adapter_images" in metadata:
        for i, ip_adapter_dict in enumerate(metadata["ip_adapter_images"]):
            ip_adapter_dict["image"] = get_image_metadata(ip_adapter_dict["image"])
    if "layers" in metadata:
        for layer in metadata["layers"]:
            redact_images_from_metadata(layer)
