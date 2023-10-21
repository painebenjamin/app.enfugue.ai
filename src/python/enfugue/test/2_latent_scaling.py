import os
import torch

from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.manager import DiffusionPipelineManager
from enfugue.diffusion.util import LatentScaler, MaskWeightBuilder
from enfugue.util import image_from_uri

UPSCALE_SOURCE="https://worldtoptop.com/wp-content/uploads/2014/04/cheam_field_tulips_agassiz1.jpg"
DOWNSCALE_SOURCE="https://images.pexels.com/photos/17565900/pexels-photo-17565900/free-photo-of-reflection-of-neoclassical-building.jpeg"

def main() -> None:
    with DebugUnifiedLoggingContext():
        with torch.no_grad():
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "latent-scaling")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            source = image_from_uri(UPSCALE_SOURCE)
            downscale_source = image_from_uri(DOWNSCALE_SOURCE)

            manager = DiffusionPipelineManager()
            manager.chunking_size = 0

            device = manager.device
            dtype = manager.dtype
            pipeline = manager.pipeline

            with MaskWeightBuilder(device=device, dtype=dtype) as weight_builder:
                processed = pipeline.image_processor.preprocess(source).to(
                    device,
                    dtype=dtype
                )
                encoded = pipeline.encode_image(
                    processed,
                    weight_builder=weight_builder,
                    device=device,
                    dtype=dtype
                )
                processed_downscale = pipeline.image_processor.preprocess(downscale_source).to(
                    device,
                    dtype=dtype
                )
                encoded_downscale = pipeline.encode_image(
                    processed_downscale,
                    weight_builder=weight_builder,
                    device=device,
                    dtype=dtype
                )
                def save_decoded(encoded, name) -> None:
                    decoded = pipeline.decode_latents(
                        encoded,
                        weight_builder=weight_builder,
                        device=device
                    )
                    decoded = pipeline.denormalize_latents(decoded)
                    decoded = pipeline.image_processor.pt_to_numpy(decoded)
                    decoded = pipeline.image_processor.numpy_to_pil(decoded)
                    decoded[0].save(os.path.join(save_dir, name))

                save_decoded(encoded, "upscale-baseline.png")
                save_decoded(encoded_downscale, "downscale-baseline.png")

                scaler = LatentScaler()

                for downscale_mode in ["nearest-exact", "bilinear", "bicubic", "area", "pool-max", "pool-avg"]:
                    scaler.downscale_mode = downscale_mode
                    scaler.downscale_antialias = False

                    scaled = scaler(encoded_downscale, 0.5)
                    save_decoded(scaled, f"downscale-{downscale_mode}.png")

                    scaled = scaler(scaled, 0.5)
                    save_decoded(scaled, f"downscale-2x-{downscale_mode}.png")

                    if downscale_mode in ["bilinear", "bicubic"]:
                        scaler.downscale_antialias = True

                        scaled = scaler(encoded_downscale, 0.5)
                        save_decoded(scaled, f"downscale-{downscale_mode}-aa.png")

                        scaled = scaler(scaled, 0.5)
                        save_decoded(scaled, f"downscale-2x-{downscale_mode}-aa.png")

                for upscale_mode in ["nearest-exact", "bilinear", "bicubic"]:
                    scaler.upscale_mode = upscale_mode
                    scaler.upscale_antialias = False

                    scaled = scaler(encoded, 2.0)
                    save_decoded(scaled, f"upscale-{upscale_mode}.png")

                    scaled = scaler(scaled, 2.0)
                    save_decoded(scaled, f"upscale-2x-{upscale_mode}.png")

                    if upscale_mode in ["bilinear", "bicubic"]:
                        scaler.upscale_antialias = True

                        scaled = scaler(encoded, 2.0)
                        save_decoded(scaled, f"upscale-{upscale_mode}-aa.png")

                        scaled = scaler(scaled, 2.0)
                        save_decoded(scaled, f"upscale-2x-{upscale_mode}-aa.png")

if __name__ == "__main__":
    main()
