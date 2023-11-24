from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.util import profiler

from datetime import datetime

def test_pipeline() -> None:
    manager = DiffusionPipelineManager()
    manager.model = "photon_v1.safetensors"
    manager.scheduler = "unipc"
    manager.seed = 3673353292

    kwargs = {
        "prompt": "Gordon Freeman from Half Life wearing metalic breastplate and standing in a futuristic laboratory, old man, medium shot, bokeh, wide angle",
        "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:1.8)",
        "num_inference_steps": 10,
        "guidance_scale": 3.1,
        "width": 680,
        "height": 1024,
        "tiling_stride": 0,
    }

    manager(**kwargs)
    manager.seed = 3673353292
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
    manager.model = "photon_v1.safetensors"
    manager.scheduler = "unipc"
    manager.seed = 3673353292

    kwargs = {
        "prompt": "Gordon Freeman from Half Life wearing metalic breastplate and standing in a futuristic laboratory, old man, medium shot, bokeh, wide angle",
        "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:1.8)",
        "num_inference_steps": 10,
        "guidance_scale": 3.1,
        "width": 680,
        "height": 1024,
        "tiling_stride": 0,
    }

    manager(**kwargs)
    manager.seed = 3673353292
    print("First image made, testing.")
    kwargs["num_results_per_prompt"] = 4
    with profiler():
        for i in range(4):
            for j, image in enumerate(manager(**kwargs)["images"]):
                image.save(f"{i}-{j}.png")

def test_engine() -> None:
    with DiffusionEngine() as engine:
        kwargs = {
            "prompt": "Gordon Freeman from Half Life wearing metalic breastplate and standing in a futuristic laboratory, old man, medium shot, bokeh, wide angle",
            "negative_prompt": "cartoon, painting, illustration, (worst quality, low quality, normal quality:1.8)",
            "num_inference_steps": 10,
            "guidance_scale": 3.1,
            "width": 680,
            "height": 1024,
            "tiling_stride": 0,
            "model": "photon_v1.safetensors",
            "scheduler": "unipc",
            "seed": 3673353292
        }

        engine(**kwargs)
        print("First image made, testing.")
        kwargs["num_results_per_prompt"] = 4
        with profiler():
            for i in range(4):
                for j, image in enumerate(engine(**kwargs)["images"]):
                    image.save(f"{i}-{j}.png")

if __name__ == "__main__":
    #test_pipeline()
    test_manager()
    #test_engine()
