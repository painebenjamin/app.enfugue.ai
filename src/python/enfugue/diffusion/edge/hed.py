# type: ignore
# Adapted from https://github.com/ashukid/hed-edge-detector/blob/master/edge.py

from typing import Any, Sequence, Tuple, List

__all__ = ["HEDCropLayer"]


class HEDCropLayer:
    """
    Added to the CV2 NN during HED edge detection.
    """

    def __init__(self, *args: Any) -> None:
        self.x_start, self.x_end = 0, 0
        self.y_start, self.y_end = 0, 0

    def getMemoryShapes(self, inputs: Sequence[Tuple[int, ...]]) -> List[List[int]]:
        """
        Gets the shapes from the input layer.
        We received two inputs. We need to crop the first blob to match the shape of the second.
        """
        input_shape, target_shape = inputs[0:2]
        batch_size, num_channels, input_height, input_width = input_shape[0:4]
        target_height, target_width = target_shape[2:4]

        self.y_start = (input_height - target_height) // 2
        self.x_start = (input_width - target_width) // 2
        self.y_end = self.y_start + target_height
        self.x_end = self.x_start + target_width

        return [[batch_size, num_channels, target_height, target_width]]

    def forward(self, inputs: Sequence[Sequence[Sequence]]) -> List[Sequence]:
        """
        On forward only send first input blob.
        """
        return [inputs[0][:, :, self.y_start : self.y_end, self.x_start : self.x_end]]
