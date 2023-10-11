from __future__ import annotations

from typing import Dict, Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor, dtype as DType

__all__ = ["DTypeConverter", "tensor_to_image"]

def tensor_to_image(latents: Tensor) -> Image:
    """
    Converts tensor to pixels using torchvision.
    """
    import torch
    from torchvision.utils import make_grid
    from PIL import Image
    grid = make_grid(latents)
    return Image.fromarray(
        grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
    )

class DTypeConverter:
    """
    This class converts between numpy and torch types.
    """
    @staticmethod
    def from_torch(torch_type: DType) -> Type:
        """
        Gets the numpy type from torch.
        :raises: KeyError When type is unknown.
        """
        import torch
        import numpy as np
        torch_to_numpy: Dict[torch.dtype, Type] = {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
            torch.bool: np.bool_,
        }
        return torch_to_numpy[torch_type]

    @staticmethod
    def from_numpy(numpy_type: Type) -> DType:
        """
        Gets the torch type from nump.
        :raises: KeyError When type is unknown.
        """
        import torch
        import numpy as np
        numpy_to_torch: Dict[Type, torch.dtype] = {
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128,
            np.bool_: torch.bool,
        }
        return numpy_to_torch[numpy_type]

    @staticmethod
    def __call__(type_to_convert: Union[DType, Type]) -> Union[DType, Type]:
        """
        Converts from one type to the other, inferring based on the type passed.
        """
        import torch
        if isinstance(type_to_convert, torch.dtype):
            return DTypeConverter.from_torch(type_to_convert)
        return DTypeConverter.from_numpy(type_to_convert)
