import sys

from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.util import profiler

from datetime import datetime

PROMPT = "Gordon Freeman from Half Life wearing metalic breastplate and standing in a futuristic laboratory, old man, medium shot, bokeh, wide angle"
NEGATIVE_PROMPT = "cartoon, painting, illustration, (worst quality, low quality, normal quality:1.8)"
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 3.1
WIDTH = 680
HEIGHT = 1024
MODEL = "epicrealism_pureEvolutionV5.safetensors"
SCHEDULER = "unipc"
SEED = 367335329287

def test_pipeline() -> None:
    manager = DiffusionPipelineManager()
    manager.model = MODEL
    manager.scheduler = SCHEDULER
    manager.seed = SEED
    manager.safe = False

    kwargs = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "tiling_stride": 0,
    }

    manager(**kwargs)
    manager.seed = SEED
    print("First image made, testing.")
    kwargs["num_results_per_prompt"] = 4
    with profiler():
        for i in range(4):
            result = manager.pipeline(
                device=manager.device,
                generator=manager.generator,
                noise_generator=manager.noise_generator,
                **kwargs
            )
            for j, image in enumerate(result["images"]):
                image.save(f"{i}-{j}.png")

def test_manager() -> None:
    manager = DiffusionPipelineManager()
    manager.model = MODEL
    manager.scheduler = SCHEDULER
    manager.seed = SEED
    manager.safe = False

    kwargs = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "tiling_stride": 0,
    }

    manager(**kwargs)
    manager.seed = SEED
    print("First image made, testing.")
    kwargs["num_results_per_prompt"] = 4
    with profiler():
        for i in range(4):
            for j, image in enumerate(manager(**kwargs)["images"]):
                image.save(f"{i}-{j}.png")

def test_engine() -> None:
    with DiffusionEngine() as engine:
        kwargs = {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "width": WIDTH,
            "height": HEIGHT,
            "tiling_stride": 0,
            "model": MODEL,
            "scheduler": SCHEDULER,
            "seed": SEED,
        }

        engine(**kwargs)
        print("First image made, testing.")
        kwargs["num_results_per_prompt"] = 4
        with profiler():
            for i in range(4):
                for j, image in enumerate(engine(**kwargs)["images"]):
                    image.save(f"{i}-{j}.png")

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "pipeline":
        test_pipeline()
    elif which == "manager":
        test_manager()
    elif which == "engine":
        test_engine()
