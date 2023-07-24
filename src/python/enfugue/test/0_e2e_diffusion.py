"""
Runs an end-to-end test of an active installation.
"""
import os

from pibble.util.log import DebugUnifiedLoggingContext

from enfugue.diffusion.support.vision import ComputerVision
from enfugue.diffusion.constants import DEFAULT_SDXL_MODEL, DEFAULT_SDXL_REFINER
from enfugue.client import EnfugueClient
from enfugue.util import logger, fit_image, image_from_uri
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
from typing import Any, List

GRID_SIZE = 256
GRID_COLS = 4
CAPTION_HEIGHT = 50
CHECKPOINT = "realisticVisionV20_v20.ckpt"
CHECKPOINT_URL = "https://civitai.com/api/download/models/29460?type=Model&format=SafeTensor&size=pruned&fp=fp16"
INPAINT_IMAGE = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
INPAINT_MASK = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"

TEST_MODE = os.getenv("ENFUGUE_TEST_MODE", "")
TEST_MODE_SAVE = "show" not in TEST_MODE.lower()
TEST_MODE_SHOW = not TEST_MODE_SAVE
TEST_MODE_SHOW_GRID = not TEST_MODE_SAVE and "grid" in TEST_MODE.lower()

def split_text(text: str, maxlen: int = 40) -> str:
    """
    Splits text into lines based on max length.
    """
    lines = 1 + len(text) // maxlen
    return "\n".join([
        text[(i*maxlen):((i+1)*maxlen)]
        for i in range(lines)
    ])

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images", "e2e")
        client = EnfugueClient()
        client.configure(
            client = {
                "host": "app.enfugue.ai",
                "port": 45554,
                "secure": True
            }
        )

        all_results = OrderedDict()

        def save_results(name: str, results: List[Image.Image]) -> List[Image.Image]:
            nonlocal all_results
            for i, result in enumerate(results):
                if TEST_MODE_SHOW and not TEST_MODE_SHOW_GRID:
                    ComputerVision.show(f"Result for \"{name}\" sample {i+1}", result)
                elif TEST_MODE_SAVE:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    result_path = os.path.join(save_dir, f"{name}_{i}.png")
                    result.save(result_path)
                    logger.info(f"Saved result for \"{name}\" sample {i+1} to {result_path}")
            all_results[name] = results
            return results

        def invoke(name: str, **kwargs: Any) -> List[Image.Image]:
            nonlocal all_results
            results = []
            kwargs["seed"] = 987654321
            if "model" not in kwargs:
                kwargs["model"] = CHECKPOINT
            kwargs["intermediates"] = False
            logger.info(f"Testing {name}\n{kwargs}")
            result = client.invoke(**kwargs)
            images = result.results()
            result.delete()
            return save_results(name, images)

        gpu_name = "Unknown GPU"
        status = client.status()
        if "gpu" in status:
            gpu = status["gpu"]
            gpu_name = gpu.get("name", gpu_name)
        
        logger.info(f"Starting e2e test on {gpu_name}")
        checkpoints = client.checkpoints()
        if CHECKPOINT not in checkpoints:
            logger.info(f"Downloading checkpoint {CHECKPOINT}")
            client.download("checkpoint", CHECKPOINT_URL, filename=CHECKPOINT)
        
        # Base txt2img
        prompt = "A man and woman standing outside a house, happy couple purchasing their first home"
        base = invoke("txt2img", prompt=prompt)[0]
        
        # Base img2img
        invoke(
            "img2img",
            prompt=prompt,
            nodes=[
                {
                    "image": base,
                    "infer": True
                }
            ]
        )
        
        # Base inpaint + fit
        inpaint_image = image_from_uri(INPAINT_IMAGE)
        inpaint_mask = image_from_uri(INPAINT_MASK)
        save_results("inpaint-image", [inpaint_image])
        save_results("inpaint-mask", [inpaint_mask])
        invoke(
            "inpaint", 
            prompt="a handsome man with ray-ban sunglasses",
            nodes=[
                {
                    "image": inpaint_image,
                    "mask": inpaint_mask,
                    "inpaint": True,
                    "w": 512,
                    "h": 512,
                    "fit": "cover"
                }
            ]
        )
        # Automatic background removal with no inference
        invoke(
            "background", 
            nodes=[
                {
                    "image": inpaint_image,
                    "remove_background": True,
                    "w": 512,
                    "h": 512,
                    "fit": "cover"
                }
            ]
        )
        
        # Automatic background removal with outpaint
        invoke(
            "background-fill",
            prompt="a handsome man outside on a sunny day, green forest in the distance",
            nodes=[
                {
                    "image": inpaint_image,
                    "remove_background": True,
                    "w": 512,
                    "h": 512,
                    "fit": "cover",
                }
            ]
        )
        
        # Sizing, fitting and outpaint
        invoke(
            "outpaint", 
            prompt="a handsome man outside on a boardwalk, overcast day",
            negative_prompt="frame, framing, comic book paneling, multiple images, awning, roof",
            nodes=[
                {
                    "image": inpaint_image,
                    "x": 128,
                    "y": 128,
                    "w": 256,
                    "h": 256,
                    "fit": "cover"
                }
            ]
        )

        # Regions + multi-diffusion
        invoke(
            "regions", 
            prompt="Roses in a bouquet",
            chunking_size=128,
            chunking_blur=128,
            nodes=[
                {
                    "x": 0,
                    "y": 0,
                    "w": 256,
                    "h": 512,
                    "prompt": "A single red rose, white background",
                    "negative_prompt": "bouquet",
                    "remove_background": True
                },
                {
                    "x": 256,
                    "y": 0,
                    "w": 256,
                    "h": 512,
                    "prompt": "A single white rose, black background",
                    "negative_prompt": "bouquet",
                    "remove_background": True
                }
            ]
        )
        
        # Controlnets
        for controlnet in ["canny", "hed", "pidi", "scribble", "depth", "normal", "mlsd", "line", "anime", "pose", "mlsd"]:
            invoke(f"txt2img-controlnet-{controlnet}", prompt=prompt, nodes=[{"image": base, "control": True, "controlnet": controlnet}])
            invoke(f"img2img-controlnet-{controlnet}", prompt=prompt, nodes=[{"image": base, "control": True, "infer": True, "controlnet": controlnet}])
        
        # Schedulers
        for scheduler in ["ddim", "ddpm", "dpmsm", "dpmss", "heun", "dpmd", "adpmd", "dpmsde", "unipc", "lmsd", "pndm", "eds", "eads"]:
            invoke(f"txt2img-scheduler-{scheduler}", prompt=prompt, scheduler=scheduler)
        
        # Multi Schedulers
        for scheduler in ["ddim", "ddpm", "deis", "dpmsm", "dpmss", "eds", "eads"]:
            invoke(f"txt2img-multi-scheduler-{scheduler}", prompt=prompt, multi_scheduler=scheduler, height=768)

        # Upscalers
        invoke(f"upscale-standalone-esrgan", outscale=2, upscale="esrgan", nodes=[{"image": base}])
        invoke(f"upscale-standalone-gfpgan", outscale=2, upscale="gfpgan", nodes=[{"image": base}])
        invoke(
            f"upscale-iterative-diffusion",
            prompt="A green tree frog",
            outscale=4,
            upscale="esrgan",
            upscale_iterative=True,
            upscale_diffusion=True,
            upscale_diffusion_steps=10,
            upscale_diffusion_strength=0.2,
            upscale_diffusion_controlnet="tile",
            upscale_diffusion_chunking_size=256,
            upscale_diffusion_chunking_blur=256
        )

        # SDXL
        if DEFAULT_SDXL_MODEL in checkpoints:
            invoke("sdxl", model=DEFAULT_SDXL_MODEL, prompt="A bride and groom on their wedding day", guidance_scale=6)
            if DEFAULT_SDXL_REFINER in checkpoints:
                invoke("sdxl-refined", model=DEFAULT_SDXL_MODEL, refiner=DEFAULT_SDXL_REFINER, prompt="A bride and groom on their wedding day", guidance_scale=6)

        # Make grid
        total_results = sum([len(arr) for arr in all_results.values()])
        rows = (total_results // GRID_COLS) + 1
        cols = total_results % GRID_COLS if total_results < GRID_COLS else GRID_COLS
        width = GRID_SIZE * cols
        height = (GRID_SIZE * rows) + (CAPTION_HEIGHT * rows)
        grid = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default()
        row, col = 0, 0

        for name in all_results:
            for i, image in enumerate(all_results[name]):
                width, height = image.size
                image = fit_image(image, GRID_SIZE, GRID_SIZE, "contain", "center-center")
                grid.paste(image, (col * GRID_SIZE, row * (GRID_SIZE + CAPTION_HEIGHT)))
                draw.text(
                    (col * GRID_SIZE + 5, row * (GRID_SIZE + CAPTION_HEIGHT) + GRID_SIZE + 2),
                    split_text(f"\"{name}\", sample {i+1}, {width}×{height}"),
                    fill=(0,0,0),
                    font=font
                )
                col += 1
                if col >= GRID_COLS:
                    row += 1
                    col = 0
        if TEST_MODE_SHOW:
            ComputerVision.show("Results", grid)
        elif TEST_MODE_SAVE:
            grid_path = os.path.join(save_dir, "grid.png")
            grid.save(grid_path)
            logger.info(f"Saved grid result at {grid_path}")


if __name__ == "__main__":
    main()
