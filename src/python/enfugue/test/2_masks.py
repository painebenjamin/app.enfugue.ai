import os
import math
import torch

from pibble.util.log import DebugUnifiedLoggingContext
from enfugue.diffusion.util import MaskWeightBuilder, tensor_to_image, get_optimal_device

SOURCE = "https://worldtoptop.com/wp-content/uploads/2014/04/cheam_field_tulips_agassiz1.jpg"

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "masks")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with MaskWeightBuilder(device=get_optimal_device(), dtype=torch.float16) as builder:
            for mask_type in ["bilinear", "gaussian"]:
                for i in range(0b1111 + 1):
                    left = i & 0b0001 == 1
                    top = i & 0b0010 == 2
                    right = i & 0b0100 == 4
                    bottom = i & 0b1000 == 8
                    name = mask_type
                    if left:
                        name = f"{name}-left"
                    if top:
                        name = f"{name}-top"
                    if right:
                        name = f"{name}-right"
                    if bottom:
                        name = f"{name}-bottom"
                    result = builder(
                        mask_type=mask_type,
                        batch=1,
                        dim=3,
                        width=128,
                        height=128,
                        unfeather_left=left,
                        unfeather_right=right,
                        unfeather_top=top,
                        unfeather_bottom=bottom,
                    )
                    tensor_to_image(result).save(os.path.join(save_dir, f"{name}.png"))

if __name__ == "__main__":
    main()
