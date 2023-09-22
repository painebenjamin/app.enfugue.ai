from typing import List, Tuple, Any
from hashlib import md5

__all__ = [
    "get_clip_engine_key",
    "get_unet_engine_key",
    "get_vae_engine_key",
    "get_controlled_unet_engine_key",
]

def get_clip_engine_key(
    size: int,
    lora: List[Tuple[str, float]],
    lycoris: List[Tuple[str, float]],
    inversion: List[str],
    **kwargs: Any
) -> str:
    """
    Uses hashlib to generate the unique key for the CLIP engine.
    CLIP must be rebuilt for each:
        1. Model
        2. Dimension
        3. LoRA
        4. LyCORIS
        5. Textual Inversion
    """
    return md5(
        "-".join(
            [
                str(size),
                ":".join(
                    "=".join([str(part) for part in lora_weight])
                    for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                ),
                ":".join(
                    "=".join([str(part) for part in lycoris_weight])
                    for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                ),
                ":".join(sorted(inversion)),
            ]
        ).encode("utf-8")
    ).hexdigest()
    
def get_unet_engine_key(
    size: int,
    lora: List[Tuple[str, float]],
    lycoris: List[Tuple[str, float]],
    inversion: List[str],
    **kwargs: Any,
) -> str:
    """
    Uses hashlib to generate the unique key for the UNET engine.
    UNET must be rebuilt for each:
        1. Model
        2. Dimension
        3. LoRA
        4. LyCORIS
        5. Textual Inversion
    """
    return md5(
        "-".join(
            [
                str(size),
                ":".join(
                    "=".join([str(part) for part in lora_weight])
                    for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                ),
                ":".join(
                    "=".join([str(part) for part in lycoris_weight])
                    for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                ),
                ":".join(sorted(inversion)),
            ]
        ).encode("utf-8")
    ).hexdigest()

def get_controlled_unet_key(
    size: int,
    lora: List[Tuple[str, float]],
    lycoris: List[Tuple[str, float]],
    inversion: List[str],
    **kwargs: Any,
) -> str:
    """
    Uses hashlib to generate the unique key for the UNET engine with controlnet blocks.
    ControlledUNET must be rebuilt for each:
        1. Model
        2. Dimension
        3. LoRA
        4. LyCORIS
        5. Textual Inversion
    """
    return md5(
        "-".join(
            [
                str(size),
                ":".join(
                    "=".join([str(part) for part in lora_weight])
                    for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                ),
                ":".join(
                    "=".join([str(part) for part in lycoris_weight])
                    for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                ),
                ":".join(sorted(inversion)),
            ]
        ).encode("utf-8")
    ).hexdigest()

def get_controlled_unet_engine_key(
    size: int,
    lora: List[Tuple[str, float]],
    lycoris: List[Tuple[str, float]],
    inversion: List[str],
    **kwargs: Any,
) -> str:
    """
    Uses hashlib to generate the unique key for the UNET engine with controlnet blocks.
    ControlledUNET must be rebuilt for each:
        1. Model
        2. Dimension
        3. LoRA
        4. LyCORIS
        5. Textual Inversion
    """
    return md5(
        "-".join(
            [
                str(size),
                ":".join(
                    "=".join([str(part) for part in lora_weight])
                    for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                ),
                ":".join(
                    "=".join([str(part) for part in lycoris_weight])
                    for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                ),
                ":".join(sorted(inversion)),
            ]
        ).encode("utf-8")
    ).hexdigest()

def get_vae_engine_key(
    size: int,
    **kwargs: Any,
) -> str:
    """
    Uses hashlib to generate the unique key for the VAE engine. VAE need only be rebuilt for each:
        1. Model
        2. Dimension
    """
    return md5(str(size).encode("utf-8")).hexdigest()
