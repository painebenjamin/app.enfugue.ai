from __future__ import annotations

import numpy as np

from enfugue.diffusion.support.model import SupportModel

from typing import Iterator, Any, Tuple, List, TYPE_CHECKING

from contextlib import contextmanager

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image

__all__ = ["Interpolator"]

class InterpolatorImageProcessor:
    """
    Used to interpolate between two images.
    """
    def __init__(
        self,
        model: torch.ScriptModule,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs: Any
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        left: Image.Image,
        right: Image.Image,
        num_frames: int = 1,
        include_ends: bool = False
    ) -> List[Image.Image]:
        """
        Runs the interpolator.
        """
        import torch
        import numpy as np
        import bisect
        import cv2
        from PIL import Image

        def pad_batch(batch: np.ndarray, align: int) -> Tuple[np.ndarray, List[int]]:
            """
            Bads an image batch to a size
            """
            height, width = batch.shape[1:3]
            height_to_pad = (align - height % align) if height % align != 0 else 0
            width_to_pad = (align - width % align) if width % align != 0 else 0

            crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
            batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                                   (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
            return batch, crop_region

        def convert_image(pil_image: Image.Image, align: int=64) -> Tuple[np.ndarray, List[int]]:
            """
            Converts a PIL image to a padded NP image
            """
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
            image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
            return image_batch, crop_region

        left, crop_region_1 = convert_image(left)
        right, crop_region_2 = convert_image(right)

        left = torch.from_numpy(left).permute(0, 3, 1, 2)
        right = torch.from_numpy(right).permute(0, 3, 1, 2)

        indexes = [0, num_frames + 1]
        remains = list(range(1, num_frames + 1))
        splits = torch.linspace(0, 1, num_frames + 2)
        results = [left, right]

        for i in range(len(remains)):
            starts = splits[indexes[:-1]]
            ends = splits[indexes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape) # type: ignore[arg-type]
            end_i = start_i + 1

            x0 = results[start_i].to(device=self.device, dtype=self.dtype)
            x1 = results[end_i].to(device=self.device, dtype=self.dtype)

            dt = x0.new_full(
                (1, 1),
                (splits[remains[step]] - splits[indexes[start_i]])
            ) / (splits[indexes[end_i]] - splits[indexes[start_i]])

            with torch.no_grad():
                predicted = self.model(x0, x1, dt) # type: ignore[operator]

            insert_position = bisect.bisect_left(indexes, remains[step])
            indexes.insert(insert_position, remains[step])
            results.insert(insert_position, predicted.clamp(0, 1).cpu().float())
            del remains[step]

        y1, x1, y2, x2 = crop_region_1
        results = [
            (tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy()
            for tensor in results
        ]
        result_images = [
            Image.fromarray(result) for result in results
        ]

        if not include_ends:
            return result_images[1:-1]
        return result_images

class Interpolator(SupportModel):
    """
    Used to remove backgrounds from images automatically
    """
    FILM_NET_PATH = "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp16.pt"

    @contextmanager
    def film(self) -> Iterator[InterpolatorImageProcessor]:
        """
        Instantiate the FILM interpolator and return the callable
        """
        import torch
        with self.context():
            model_path = self.get_model_file(self.FILM_NET_PATH)
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval().to(device=self.device, dtype=self.dtype)
            processor = InterpolatorImageProcessor(model, self.device, self.dtype)
            yield processor
            del processor
            del model
