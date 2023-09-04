import os
import torch

from typing import Optional
from PIL.Image import Image
from diffusers.pipelines.stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from diffusers.schedulers import *

def create_image(prompt: str, refiner_start: float, scheduler: KarrasDiffusionSchedulers) -> Image:
    """
    Creates an image using SDXL base + refiner.
    """
    pipeline = StableDiffusionXLPipeline.from_single_file(
        os.path.expanduser("~/.cache/enfugue/checkpoint/sd_xl_base_1.0.safetensors"),
    ).to("cuda")
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(12345)

    partially_denoised = pipeline(
        prompt=prompt,
        denoising_end=refiner_start,
        output_type="latent",
        generator=generator,
        num_inference_steps=25
    )["images"][0]

    del pipeline
    torch.cuda.empty_cache()

    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
        os.path.expanduser("~/.cache/enfugue/checkpoint/sd_xl_refiner_1.0.safetensors"),
    ).to("cuda")
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    result = pipeline(
        prompt=prompt,
        denoising_start=refiner_start,
        image=partially_denoised,
        output_type="pil",
        generator=generator,
        num_inference_steps=25
    )["images"][0]
    
    del pipeline
    torch.cuda.empty_cache()

    return result

prompt = "A happy-looking puppy"
refiner_start = 0.8

create_image(prompt, refiner_start, EulerDiscreteScheduler).save("./euler.png")
create_image(prompt, refiner_start, KDPM2DiscreteScheduler).save("./dpmd.png")
