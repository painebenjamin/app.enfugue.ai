from __future__ import annotations

import os

from typing import (
    Optional,
    Union,
    Tuple,
    Literal,
    List,
    TYPE_CHECKING
)
from typing_extensions import (
    TypedDict,
    NotRequired
)
from enfugue.util import (
    IMAGE_FIT_LITERAL,
    IMAGE_ANCHOR_LITERAL
)

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "UpscaleStepDict",
    "ImageDict",
    "IPAdapterImageDict",
    "ControlImageDict",
    "PromptDict",
    "DEFAULT_MODEL",
    "DEFAULT_INPAINTING_MODEL",
    "DEFAULT_SDXL_MODEL",
    "DEFAULT_SDXL_REFINER",
    "DEFAULT_SDXL_INPAINTING_MODEL",
    "VAE_EMA",
    "VAE_MSE",
    "VAE_XL",
    "VAE_XL16",
    "VAE_LITERAL",
    "CONTROLNET_CANNY",
    "CONTROLNET_CANNY_XL",
    "CONTROLNET_MLSD",
    "CONTROLNET_HED",
    "CONTROLNET_SCRIBBLE",
    "CONTROLNET_TILE",
    "CONTROLNET_INPAINT",
    "CONTROLNET_DEPTH",
    "CONTROLNET_DEPTH_XL",
    "CONTROLNET_NORMAL",
    "CONTROLNET_POSE",
    "CONTROLNET_POSE_XL",
    "CONTROLNET_PIDI",
    "CONTROLNET_PIDI_XL",
    "CONTROLNET_LINE",
    "CONTROLNET_ANIME",
    "CONTROLNET_TEMPORAL",
    "CONTROLNET_QR",
    "CONTROLNET_QR_XL",
    "CONTROLNET_LITERAL",
    "MOTION_LORA_ZOOM_IN",
    "MOTION_LORA_ZOOM_OUT",
    "MOTION_LORA_PAN_LEFT",
    "MOTION_LORA_PAN_RIGHT",
    "MOTION_LORA_ROLL_CLOCKWISE",
    "MOTION_LORA_ROLL_ANTI_CLOCKWISE",
    "MOTION_LORA_TILT_UP",
    "MOTION_LORA_TILT_DOWN",
    "MOTION_LORA_LITERAL",
    "SCHEDULER_LITERAL",
    "DEVICE_LITERAL",
    "PIPELINE_SWITCH_MODE_LITERAL",
    "UPSCALE_LITERAL",
    "MASK_TYPE_LITERAL",
    "LOOP_TYPE_LITERAL",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_INVERSION_DIR",
    "DEFAULT_LORA_DIR",
    "DEFAULT_LYCORIS_DIR",
    "DEFAULT_TENSORRT_DIR",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_OTHER_DIR",
    "DEFAULT_DIFFUSERS_DIR",
    "DEFAULT_SIZE",
    "DEFAULT_SDXL_SIZE",
    "DEFAULT_TEMPORAL_SIZE",
    "DEFAULT_TILING_SIZE",
    "DEFAULT_TILING_STRIDE",
    "DEFAULT_TEMPORAL_TILING_SIZE",
    "DEFAULT_IMAGE_CALLBACK_STEPS",
    "DEFAULT_CONDITIONING_SCALE",
    "DEFAULT_IMG2IMG_STRENGTH",
    "DEFAULT_INFERENCE_STEPS",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_UPSCALE_PROMPT",
    "DEFAULT_UPSCALE_INFERENCE_STEPS",
    "DEFAULT_UPSCALE_GUIDANCE_SCALE",
    "DEFAULT_UPSCALE_TILING_SIZE",
    "DEFAULT_UPSCALE_TILING_STRIDE",
    "DEFAULT_REFINER_START",
    "DEFAULT_REFINER_STRENGTH",
    "DEFAULT_REFINER_GUIDANCE_SCALE",
    "DEFAULT_AESTHETIC_SCORE",
    "DEFAULT_NEGATIVE_AESTHETIC_SCORE",
    "MODEL_PROMPT_WEIGHT",
    "GLOBAL_PROMPT_STEP_WEIGHT",
    "GLOBAL_PROMPT_UPSCALE_WEIGHT",
    "UPSCALE_PROMPT_STEP_WEIGHT",
    "MAX_IMAGE_SCALE",
    "LATENT_BLEND_METHOD_LITERAL",
    "NOISE_METHOD_LITERAL",
    "IP_ADAPTER_LITERAL"
]

DEFAULT_MODEL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
DEFAULT_INPAINTING_MODEL = "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
DEFAULT_SDXL_MODEL = "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/main/sd_xl_base_1.0_fp16_vae.safetensors"
DEFAULT_SDXL_REFINER = "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
DEFAULT_SDXL_INPAINTING_MODEL = "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/main/sd_xl_base_1.0_inpainting_0.1.safetensors"

DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/.cache/enfugue/checkpoint")
DEFAULT_INVERSION_DIR = os.path.expanduser("~/.cache/enfugue/inversion")
DEFAULT_TENSORRT_DIR = os.path.expanduser("~/.cache/enfugue/tensorrt")
DEFAULT_LORA_DIR = os.path.expanduser("~/.cache/enfugue/lora")
DEFAULT_LYCORIS_DIR = os.path.expanduser("~/.cache/enfugue/lycoris")
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/enfugue/cache")
DEFAULT_DIFFUSERS_DIR = os.path.expanduser("~/.cache/enfugue/diffusers")
DEFAULT_OTHER_DIR = os.path.expanduser("~/.cache/enfugue/other")

DEFAULT_SIZE = 512
DEFAULT_TILING_SIZE = 512
DEFAULT_TILING_STRIDE = 32
DEFAULT_TILING_MASK = "bilinear"
DEFAULT_TEMPORAL_SIZE = 16
DEFAULT_TEMPORAL_TILING_SIZE = 12
DEFAULT_SDXL_SIZE = 1024
DEFAULT_IMAGE_CALLBACK_STEPS = 10
DEFAULT_CONDITIONING_SCALE = 1.0
DEFAULT_IMG2IMG_STRENGTH = 0.8
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_UPSCALE_PROMPT = "highly detailed, ultra-detailed, intricate detail, high definition, HD, 4k, 8k UHD"
DEFAULT_UPSCALE_INFERENCE_STEPS = 100
DEFAULT_UPSCALE_GUIDANCE_SCALE = 12
DEFAULT_UPSCALE_TILING_STRIDE = 128
DEFAULT_UPSCALE_TILING_SIZE = DEFAULT_TILING_SIZE

DEFAULT_REFINER_START = 0.85
DEFAULT_REFINER_STRENGTH = 0.3
DEFAULT_REFINER_GUIDANCE_SCALE = 5.0
DEFAULT_AESTHETIC_SCORE = 6.0
DEFAULT_NEGATIVE_AESTHETIC_SCORE = 2.5

MODEL_PROMPT_WEIGHT = 0.2
GLOBAL_PROMPT_STEP_WEIGHT = 0.4
GLOBAL_PROMPT_UPSCALE_WEIGHT = 0.4
UPSCALE_PROMPT_STEP_WEIGHT = 0.1
MAX_IMAGE_SCALE = 3.0

CACHE_MODE_LITERAL = ["always", "xl", "tensorrt"]
VAE_LITERAL = Literal["ema", "mse", "xl", "xl16"]
DEVICE_LITERAL = Literal["cpu", "cuda", "dml", "mps"]
PIPELINE_SWITCH_MODE_LITERAL = Literal["offload", "unload"]
SCHEDULER_LITERAL = Literal[
    "ddim", "ddpm", "deis",
    "dpmsm", "dpmsms", "dpmsmk", "dpmsmka",
    "dpmss", "dpmssk", "heun",
    "dpmd", "dpmdk", "adpmd",
    "adpmdk", "dpmsde", "unipc",
    "lmsd", "lmsdk", "pndm",
    "eds", "eads"
]
UPSCALE_LITERAL = Literal[
    "esrgan", "esrganime", "gfpgan",
    "lanczos", "bilinear", "bicubic",
    "nearest"
]
CONTROLNET_LITERAL = Literal[
    "canny", "mlsd", "hed",
    "scribble", "tile", "inpaint",
    "depth", "normal", "pose",
    "pidi", "line", "anime",
    "temporal", "qr"
]
MOTION_LORA_LITERAL = [
    "pan-left", "pan-right",
    "roll-clockwise", "roll-anti-clockwise",
    "tilt-up", "tilt-down",
    "zoom-in", "zoom-out"
]
LOOP_TYPE_LITERAL = Literal[
    "loop", "reflect"
]
MASK_TYPE_LITERAL = Literal[
    "constant", "bilinear", "gaussian"
]
LATENT_BLEND_METHOD_LITERAL = Literal[
    "add", "bislerp", "cosine", "cubic",
    "difference", "inject", "lerp", "slerp",
    "exclusion", "subtract", "multiply", "overlay",
    "screen", "color_dodge", "linear_dodge", "glow",
    "pin_light", "hard_light", "linear_light", "vivid_light"
]
NOISE_METHOD_LITERAL = Literal[
    "default", "crosshatch", "simplex",
    "perlin", "brownian_fractal", "white",
    "grey", "pink", "blue", "green",
    "velvet", "violet", "random_mix"
]
IP_ADAPTER_LITERAL = Literal[
    "default", "plus", "plus-face"
]

# VAE repos/files
VAE_EMA = (
    "stabilityai/sd-vae-ft-ema",
    "vae-ft-ema-560000-ema-pruned",
    "sd-vae-ft-ema-original",
    "sd-vae-ft-ema",
)
VAE_MSE = (
    "stabilityai/sd-vae-ft-mse",
    "vae-ft-mse-840000-ema-pruned",
    "sd-vae-ft-mse-original",
    "sd-vae-ft-mse",
)
VAE_XL = (
    "stabilityai/sdxl-vae",
    "sdxl-vae",
    "sdxl_vae"
)
VAE_XL16 = (
    "madebyollin/sdxl-vae-fp16-fix",
    "sdxl-vae-fp16-fix",
    "sdxl-vae-fp16",
    "sdxl_vae_fp16_fix",
    "sdxl_vae_fp16"
)
# ControlNet repos/files
CONTROLNET_CANNY = (
    "lllyasviel/control_v11p_sd15_canny",
    "control_v11p_sd15_canny",
    "control_sd15_canny",
)
CONTROLNET_MLSD = (
    "lllyasviel/control_v11p_sd15_mlsd",
    "control_v11p_sd15_mlsd",
    "control_sd15_mlsd",
)
CONTROLNET_HED = (
    "lllyasviel/sd-controlnet-hed",
    "control_sd15_hed",
    "sd-controlnet-hed",
)
CONTROLNET_SCRIBBLE = (
    "lllyasviel/control_v11p_sd15_scribble",
    "control_v11p_sd15_scribble",
    "control_sd15_scribble",
)
CONTROLNET_TILE = (
    "lllyasviel/control_v11f1e_sd15_tile",
    "control_v11f1e_sd15_tile",
    "control_sd15_tile",
)
CONTROLNET_INPAINT = (
    "lllyasviel/control_v11p_sd15_inpaint",
    "control_v11p_sd15_inpaint",
    "control_sd15_inpaint",
)
CONTROLNET_DEPTH = (
    "lllyasviel/control_v11f1p_sd15_depth",
    "control_v11f1p_sd15_depth",
    "control_sd15_depth",
)
CONTROLNET_NORMAL = (
    "lllyasviel/control_v11p_sd15_normalbae",
    "control_v11p_sd15_normalbae",
    "control_sd15_normal",
)
CONTROLNET_POSE = (
    "lllyasviel/control_v11p_sd15_openpose",
    "control_v11p_sd15_openpose",
    "control_sd15_openpose",
)
CONTROLNET_PIDI = (
    "lllyasviel/control_v11p_sd15_softedge",
    "control_v11p_sd15_softedge",
)
CONTROLNET_PIDI_XL = (
    "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    "controlnet-sd-xl-1.0-softedge-dexined"
)
CONTROLNET_LINE = (
    "lllyasviel/control_v11p_sd15_lineart",
    "control_v11p_sd15_lineart",
)
CONTROLNET_ANIME = (
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "control_v11p_sd15s2_lineart_anime",
)
CONTROLNET_TEMPORAL = (
    "CiaraRowles/TemporalNet",
    "TemporalNet", # TODO
)
CONTROLNET_QR = (
    "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors",
    "control_v1p_sd15_qrcode_monster_v2",
    "control_v1p_sd15_qrcode_monster",
)
CONTROLNET_QR_XL = (
    "https://huggingface.co/Nacholmo/controlnet-qr-pattern-sdxl/resolve/main/automatic1111/control_v10e_sdxl_opticalpattern.safetensors",
    "control_v10e_sdxl_opticalpattern",
    "control_v04u_sdxl_opticalpattern-half",
    "control_v03u_sdxl_opticalpattern",
    "control_v02u_sdxl_qrpattern",
    "control_v02u_sdxl_opticalpattern",
    "control_v01u_sdxl_opticalpattern",
    "control_v01u_sdxl_qrpattern",
)
CONTROLNET_CANNY_XL = (
    "diffusers/controlnet-canny-sdxl-1.0",
    "diffusers_xl_canny_full",
    "controlnet-canny-sdxl-1.0",
)
CONTROLNET_DEPTH_XL = (
    "diffusers/controlnet-depth-sdxl-1.0",
    "diffusers_xl_depth_full",
    "controlnet-depth-sdxl-1.0",
)
CONTROLNET_POSE_XL = (
    "thibaud/controlnet-openpose-sdxl-1.0",
    "OpenPoseXL2",
    "controlnet-openpose-sdxl-1.0",
)

MOTION_LORA_PAN_LEFT = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt"
MOTION_LORA_PAN_RIGHT = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt"
MOTION_LORA_ROLL_CLOCKWISE = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt"
MOTION_LORA_ROLL_ANTI_CLOCKWISE = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt"
MOTION_LORA_TILT_UP = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.ckpt"
MOTION_LORA_TILT_DOWN = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.ckpt"
MOTION_LORA_ZOOM_IN = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt"
MOTION_LORA_ZOOM_OUT = "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt"

MultiModelType = Union[str, List[str]]
WeightedMultiModelType = Union[
    str, Tuple[str, float],
    List[Union[str, Tuple[str, float]]]
]

class ImageDict(TypedDict):
    """
    An image or video with optional fitting details
    """
    image: Union[str, Image, List[Image]]
    skip_frames: NotRequired[Optional[int]]
    divide_frames: NotRequired[Optional[int]]
    fit: NotRequired[Optional[IMAGE_FIT_LITERAL]]
    anchor: NotRequired[Optional[IMAGE_ANCHOR_LITERAL]]
    invert: NotRequired[bool]

class ControlImageDict(ImageDict):
    """
    Extends the image dict additionally with controlnet details
    """
    controlnet: CONTROLNET_LITERAL
    scale: NotRequired[float]
    start: NotRequired[Optional[float]]
    end: NotRequired[Optional[float]]
    process: NotRequired[bool]

class IPAdapterImageDict(ImageDict):
    """
    Extends the image dict additionally with IP adapter scale
    """
    scale: NotRequired[float]

class UpscaleStepDict(TypedDict):
    """
    All the options for each upscale step
    """
    method: UPSCALE_LITERAL
    amount: Union[int, float]
    strength: NotRequired[float]
    num_inference_steps: NotRequired[int]
    scheduler: NotRequired[SCHEDULER_LITERAL]
    guidance_scale: NotRequired[float]
    controlnets: NotRequired[List[Union[CONTROLNET_LITERAL, Tuple[CONTROLNET_LITERAL, float]]]]
    prompt: NotRequired[str]
    prompt_2: NotRequired[str]
    negative_prompt: NotRequired[str]
    negative_prompt_2: NotRequired[str]
    chunking_size: NotRequired[Optional[int]]
    chunking_frames: NotRequired[Optional[int]]
    chunking_mask: NotRequired[MASK_TYPE_LITERAL]

class PromptDict(TypedDict):
    """
    A prompt step, optionally with frame details
    """
    positive: str
    negative: NotRequired[Optional[str]]
    positive_2: NotRequired[Optional[str]]
    negative_2: NotRequired[Optional[str]]
    weight: NotRequired[Optional[float]]
    start: NotRequired[Optional[int]]
    end: NotRequired[Optional[int]]
