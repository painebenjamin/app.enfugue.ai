from __future__ import annotations

import io
import math

from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

from pibble.resources.retriever import Retriever

__all__ = [
    "fit_image",
    "feather_mask",
    "image_from_uri",
    "images_are_equal",
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
    image: Image,
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
) -> Image:
    """
    Given an image of unknown size, make it a known size with optional fit parameters.
    """
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
        blank_image.paste(image, (left, top))

        return blank_image
    elif fit == "contain":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width, horizontal_image_height = int(image_width * width_ratio), int(
            image_height * width_ratio
        )
        vertical_image_width, vertical_image_height = int(image_width * height_ratio), int(image_height * height_ratio)
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
        blank_image = Image.new("RGBA", (width, height))
        blank_image.paste(input_image, (left, top))

        return blank_image
    elif fit == "cover":
        image_width, image_height = image.size
        width_ratio, height_ratio = width / image_width, height / image_height
        horizontal_image_width, horizontal_image_height = math.ceil(image_width * width_ratio), math.ceil(
            image_height * width_ratio
        )
        vertical_image_width, vertical_image_height = math.ceil(image_width * height_ratio), math.ceil(
            image_height * height_ratio
        )
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
        blank_image = Image.new("RGBA", (width, height))
        blank_image.paste(input_image, (left, top))

        return blank_image
    elif fit == "stretch":
        return image.resize((width, height)).convert("RGBA")
    else:
        raise ValueError(f"Unknown fit {fit}")


def feather_mask(image: Image) -> Image:
    """
    Given an image, create a feathered binarized mask by 'growing' the black/white pixel sections.
    """
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
