import io
import os
import PIL
import torch
import requests

from einops import rearrange
from pibble.util.log import DebugUnifiedLoggingContext


def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test-results", "motion-vector"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from enfugue.diffusion.util import Video
        from enfugue.diffusion.util.torch_util import (
            motion_vector_conditioning_tensor,
            tensor_to_image,
        )

        tensor = motion_vector_conditioning_tensor(
            width=1024,
            height=576,
            frames=80,
            device="cuda",
            gaussian_sigma=20,
            motion_vectors=[
                [
                    {"anchor": [811, 357]},
                    {
                        "anchor": [896, 306],
                        "control_1": [883, 366],
                        "control_2": [909, 193],
                    },
                    {
                        "anchor": [785, 198],
                        "control_1": [846, 140],
                        "control_2": [751, 245],
                    },
                    {"anchor": [835, 310]},
                ],
                [
                    {"anchor": [940, 241]},
                    {
                        "anchor": [913, 120],
                        "control_1": [1003, 179],
                        "control_2": [876, 92],
                    },
                    {
                        "anchor": [839, 44],
                        "control_1": [781, 91],
                        "control_2": [875, 17],
                    },
                    {
                        "anchor": [969, 46],
                        "control_1": [933, 9],
                        "control_2": [1001, 113],
                    },
                    {"anchor": [957, 125]},
                ],
                [{"anchor": [778, 425]}, {"anchor": [676, 459]}],
                [{"anchor": [591, 431]}, {"anchor": [637, 566]}],
                [{"anchor": [682, 406]}, {"anchor": [617, 422]}],
                [{"anchor": [771, 513]}, {"anchor": [812, 571]}],
                [{"anchor": [579, 19]}, {"anchor": [510, 11]}],
                [{"anchor": [568, 239]}, {"anchor": [523, 230]}],
                [{"anchor": [509, 436]}, {"anchor": [465, 422]}],
                [{"anchor": [352, 216]}, {"anchor": [331, 213]}],
                [{"anchor": [591, 398]}, {"anchor": [539, 390]}],
            ],
        )
        motion_min = torch.min(tensor)
        motion_max = torch.max(tensor)
        tensor = (tensor - motion_min) / (motion_max - motion_min)
        f, h, w, c = tensor.shape
        tensor = torch.cat([tensor[:, :, :, 1:2].clone(), tensor], dim=3)
        Video(
            [
                tensor_to_image(rearrange(tensor[i], "h w c -> c h w"))
                for i in range(tensor.shape[0])
            ]
        ).save(os.path.join(save_dir, "motion.gif"), overwrite=True, rate=20.0)


if __name__ == "__main__":
    main()
