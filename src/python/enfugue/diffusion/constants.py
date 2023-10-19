from typing import Literal

__all__ = [
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
    "CONTROLNET_LINE",
    "CONTROLNET_ANIME",
    "CONTROLNET_LITERAL",
    "CONTROLNET_TEMPORAL",
    "CONTROLNET_QR",
    "SCHEDULER_LITERAL",
    "DEVICE_LITERAL",
    "PIPELINE_SWITCH_MODE_LITERAL",
    "UPSCALE_LITERAL",
    "MASK_TYPE_LITERAL",
    "LATENT_BLEND_METHOD_LITERAL",
    "NOISE_METHOD_LITERAL"
]

DEFAULT_MODEL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
DEFAULT_INPAINTING_MODEL = "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
DEFAULT_SDXL_MODEL = "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/main/sd_xl_base_1.0_fp16_vae.safetensors"
DEFAULT_SDXL_REFINER = "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
DEFAULT_SDXL_INPAINTING_MODEL = "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/main/sd_xl_base_1.0_inpainting_0.1.safetensors"

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

VAE_LITERAL = Literal["ema", "mse", "xl", "xl16"]

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
# Recommend XL files come from https://huggingface.co/lllyasviel/sd_control_collection/tree/main
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
CONTROLNET_LITERAL = Literal[
    "canny", "mlsd", "hed",
    "scribble", "tile", "inpaint",
    "depth", "normal", "pose",
    "pidi", "line", "anime",
    "temporal", "qr"
]

SCHEDULER_LITERAL = Literal[
    "ddim", "ddpm", "deis",
    "dpmsm", "dpmsmk", "dpmsmka",
    "dpmss", "dpmssk", "heun",
    "dpmd", "dpmdk", "adpmd",
    "adpmdk", "dpmsde", "unipc",
    "lmsd", "lmsdk", "pndm",
    "eds", "eads"
]

DEVICE_LITERAL = Literal["cpu", "cuda", "dml", "mps"]

PIPELINE_SWITCH_MODE_LITERAL = Literal["offload", "unload"]

UPSCALE_LITERAL = Literal["esrgan", "esrganime", "gfpgan", "lanczos", "bilinear", "bicubic", "nearest"]

MASK_TYPE_LITERAL = Literal["constant", "bilinear", "gaussian"]

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
