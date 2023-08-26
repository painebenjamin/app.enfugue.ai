from typing import Literal

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_INPAINTING_MODEL",
    "DEFAULT_SDXL_MODEL",
    "DEFAULT_SDXL_REFINER",
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
    "CONTROLNET_PIDI",
    "CONTROLNET_LINE",
    "CONTROLNET_ANIME",
    "CONTROLNET_LITERAL",
    "SCHEDULER_LITERAL",
    "DEVICE_LITERAL",
    "PIPELINE_SWITCH_MODE_LITERAL",
    "UPSCALE_LITERAL",
    "UPSCALE_PIPELINE_LITERAL"
]

DEFAULT_MODEL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
DEFAULT_INPAINTING_MODEL = (
    "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
)
DEFAULT_SDXL_MODEL = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
DEFAULT_SDXL_REFINER = "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"

VAE_EMA = "stabilityai/sd-vae-ft-ema"
VAE_MSE = "stabilityai/sd-vae-ft-mse"
VAE_XL = "stabilityai/sdxl-vae"
VAE_XL16 = "madebyollin/sdxl-vae-fp16-fix"

VAE_LITERAL = Literal["ema", "mse", "xl", "xl16"]

CONTROLNET_CANNY = "lllyasviel/sd-controlnet-canny"
CONTROLNET_MLSD = "lllyasviel/control_v11p_sd15_mlsd"
CONTROLNET_HED = "lllyasviel/sd-controlnet-hed"
CONTROLNET_SCRIBBLE = "lllyasviel/control_v11p_sd15_scribble"
CONTROLNET_TILE = "lllyasviel/control_v11f1e_sd15_tile"
CONTROLNET_INPAINT = "lllyasviel/control_v11p_sd15_inpaint"
CONTROLNET_DEPTH = "lllyasviel/sd-controlnet-depth"
CONTROLNET_NORMAL = "lllyasviel/sd-controlnet-normal"
CONTROLNET_POSE = "lllyasviel/control_v11p_sd15_openpose"
CONTROLNET_PIDI = "lllyasviel/control_v11p_sd15_softedge"
CONTROLNET_LINE = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
CONTROLNET_ANIME = "lllyasviel/control_v11p_sd15s2_lineart_anime"

CONTROLNET_CANNY_XL = "diffusers/controlnet-canny-sdxl-1.0"
CONTROLNET_DEPTH_XL = "diffusers/controlnet-depth-sdxl-1.0"

CONTROLNET_LITERAL = Literal[
    "canny", "mlsd", "hed", "scribble", "tile", "inpaint", "depth", "normal", "pose", "pidi", "line", "anime"
]

SCHEDULER_LITERAL = Literal[
    "ddim", "ddpm", "deis", "dpmsm", "dpmss", "heun", "dpmd", "adpmd", "dpmsde", "unipc", "lmsd", "pndm", "eds", "eads"
]
DEVICE_LITERAL = Literal["cpu", "cuda", "dml", "mps"]
PIPELINE_SWITCH_MODE_LITERAL = Literal["offload", "unload"]
UPSCALE_LITERAL = Literal["esrgan", "esrganime", "gfpgan", "lanczos", "bilinear", "bicubic", "nearest"]
UPSCALE_PIPELINE_LITERAL = Literal["base"]
