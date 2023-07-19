__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_INPAINTING_MODEL",
    "VAE_EMA",
    "VAE_MSE",
    "VAE_XL",
    "CONTROLNET_CANNY",
    "CONTROLNET_MLSD",
    "CONTROLNET_HED",
    "CONTROLNET_SCRIBBLE",
    "CONTROLNET_TILE",
    "CONTR0LNET_INPAINT",
    "CONTROLNET_DEPTH",
    "CONTROLNET_NORMAL",
    "CONTROLNET_POSE",
]

DEFAULT_MODEL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
DEFAULT_INPAINTING_MODEL = (
    "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
)

VAE_EMA = "stabilityai/sd-vae-ft-ema"
VAE_MSE = "stabilityai/sd-vae-ft-mse"
VAE_XL = "stabilityai/sdxl-vae"

CONTROLNET_CANNY = "lllyasviel/sd-controlnet-canny"
CONTROLNET_MLSD = "lllyasviel/control_v11p_sd15_mlsd"
CONTROLNET_HED = "lllyasviel/sd-controlnet-hed"
CONTROLNET_SCRIBBLE = "lllyasviel/control_v11p_sd15_scribble"
CONTROLNET_TILE = "lllyasviel/control_v11f1e_sd15_tile"
CONTROLNET_INPAINT = "lllyasviel/control_v11p_sd15_inpaint"
CONTROLNET_DEPTH = "lllyasviel/sd-controlnet-depth"
CONTROLNET_NORMAL = "lllyasviel/sd-controlnet-normal"
CONTROLNET_POSE = "lllyasviel/control_v11p_sd15_openpose"
