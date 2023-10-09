import os
import math
import torch

from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import NoiseMaker, Noiser, make_noise

SOURCE = "https://worldtoptop.com/wp-content/uploads/2014/04/cheam_field_tulips_agassiz1.jpg"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "noise")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        manager.seed = 12345
        manager.chunking_size = 0
        device = manager.device
        dtype = manager.dtype

        for noise_method in [
            "default", "crosshatch", "simplex",
            "brownian_fractal", "white",
            "grey", "pink", "blue", "green",
            "velvet", "violet", "random_mix"
        ]:
            noise_latents = make_noise(
                method=noise_method,
                width=512 // 8,
                height=512 // 8,
                channels=3,
                batch_size=1,
                generator=manager.noise_generator,
                device=manager.device,
                dtype=manager.dtype,
            )
            NoiseMaker.to_image(noise_latents).save(os.path.join(save_dir, f"./{noise_method}.png"))

if __name__ == "__main__":
    main()
