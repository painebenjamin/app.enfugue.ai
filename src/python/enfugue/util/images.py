from __future__ import annotations

import math
import io
import PIL
import PIL.Image

from typing import Optional, Literal

from pibble.resources.retriever import Retriever

__all__ = [
    "fit_image",
    "feather_mask",
    "remove_background",
    "image_from_uri",
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
    image: PIL.Image.Image,
    width: int,
    height: int,
    fit: Optional[IMAGE_FIT_LITERAL] = None,
    anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
) -> PIL.Image.Image:
    """
    Given an image of unknown size, make it a known size with optional fit parameters.
    """
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

        blank_image = PIL.Image.new("RGBA", (width, height), (0, 0, 0, 0))
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
        blank_image = PIL.Image.new("RGBA", (width, height))
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
        blank_image = PIL.Image.new("RGBA", (width, height))
        blank_image.paste(input_image, (left, top))

        return blank_image
    elif fit == "stretch":
        return image.resize((width, height)).convert("RGBA")
    else:
        raise ValueError(f"Unknown fit {fit}")


def feather_mask(image: PIL.Image.Image) -> PIL.Image.Image:
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


def remove_background(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Remove the background from an image.
    """
    import backgroundremover.utilities

    # We have to import this in this order for backgroundremover to work
    backgroundremover.utilities  # silence importchecker
    import backgroundremover.bg

    buf = io.BytesIO()
    image.save(buf, "PNG")
    return PIL.Image.open(io.BytesIO(backgroundremover.bg.remove(buf.getvalue())))


def image_from_uri(uri: str) -> PIL.Image.Image:
    """
    Loads an image using the pibble reteiever; works with http, file, ftp, ftps, sftp, and s3
    """
    return PIL.Image.open(io.BytesIO(Retriever.get(uri).all()))
