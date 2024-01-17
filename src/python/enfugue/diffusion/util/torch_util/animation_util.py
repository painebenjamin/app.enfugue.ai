from __future__ import annotations

from typing import Optional, Union, Tuple, List, TYPE_CHECKING
from enfugue.diffusion.constants import MotionVectorPointDict

if TYPE_CHECKING:
    import numpy.ndarray as NDArray
    from torch import (
        Tensor,
        device as Device,
        dtype as DType
    )
    from torch.nn import Conv2d
    from PIL.Image import Image

__all__ = [
    "linear_length",
    "linear_point",
    "quadratic_bezier_length",
    "quadratic_bezier_point",
    "cubic_bezier_length",
    "cubic_bezier_point",
    "get_segment_length",
    "get_point_along_vector",
    "get_gaussian_kernel",
    "apply_gaussian_kernel",
    "motion_vector_conditioning_tensor",
    "optical_flow_conditioning_tensor",
    "flow_condition_to_image_sequence"
]

def linear_length(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float]
) -> float:
    """
    Calculates the length between two points
    """
    import numpy as np
    # Extract x and y coordinates for each point
    x1, y1 = point_1
    x2, y2 = point_2
    dx, dy = x2-x1, y2-y1
    return np.sqrt(dx*dx+dy*dy)

def linear_point(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    t: float
) -> Tuple[int, int]:
    """
    Calculates a point along the line given t=[0,1]
    """
    import numpy as np
    # Extract x and y coordinates for each point
    x1, y1 = point_1
    x2, y2 = point_2
    dx, dy = x2-x1, y2-y1
    return int(x1+t*dx), int(y1+t*dy)

def quadratic_bezier_length(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    point_3: Tuple[float, float],
    num_points: int=1000 # accuracy
) -> float:
    """
    Calculates the length of a quadratic bezier
    """
    import numpy as np
    from scipy.integrate import quad

    def quadratic_function(
        t: float,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        x_derivative = 2 * (1 - t) * (p2[0] - p1[0]) + 2 * t * (p3[0] - p2[0])
        y_derivative = 2 * (1 - t) * (p2[1] - p1[1]) + 2 * t * (p3[1] - p2[1])
        return np.sqrt(x_derivative**2 + y_derivative**2)

    return quad(quadratic_function, 0, 1, args=(point_1, point_2, point_3))[0]

def quadratic_bezier_point(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    point_3: Tuple[float, float],
    t: float
) -> Tuple[int, int]:
    """
    Calculate a point on a quadratic bezier
    """
    # Extract x and y coordinates for each point
    x1, y1 = point_1
    x2, y2 = point_2
    x3, y3 = point_3
    x = (1 - t)**2 * x1 + 2 * (1 - t) * t * x2 + t**2 * x3
    y = (1 - t)**2 * y1 + 2 * (1 - t) * t * y2 + t**2 * y3
    return int(x), int(y)

def cubic_bezier_length(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    point_3: Tuple[float, float],
    point_4: Tuple[float, float]
) -> float:
    """
    Calculates the length of a cubic bezier
    """
    from scipy.integrate import quad
    import numpy as np
    # Extract x and y coordinates for each point
    x1, y1 = point_1
    x2, y2 = point_2
    x3, y3 = point_3
    x4, y4 = point_4

    # Define the cubic Bezier function in the form ax^3 + bx^2 + cx + d
    a = -x1 + 3*x2 - 3*x3 + x4
    b = 3*x1 - 6*x2 + 3*x3
    c = -3*x1 + 3*x2
    d = x1

    # Define the derivative of the cubic Bezier function
    def derivative(t):
        return 3*a*t**2 + 2*b*t + c

    # Calculate the length of the curve using numerical integration
    def integrand(t):
        return np.sqrt(1 + derivative(t)**2)

    # Use the quad function from scipy to perform the integration
    length, _ = quad(integrand, 0, 1)

    return length

def cubic_bezier_point(
    point_1: Tuple[float, float],
    point_2: Tuple[float, float],
    point_3: Tuple[float, float],
    point_4: Tuple[float, float],
    t: float
) -> Tuple[int, int]:
    """
    Calculate a point on a cubic bezier
    """
    # Extract x and y coordinates for each point
    x1, y1 = point_1
    x2, y2 = point_2
    x3, y3 = point_3
    x4, y4 = point_4
    # Calculate cubic bezier
    x = (1 - t)**3 * x1 + 3 * (1 - t)**2 * t * x2 + 3 * (1 - t) * t**2 * x3 + t**3 * x4
    y = (1 - t)**3 * y1 + 3 * (1 - t)**2 * t * y2 + 3 * (1 - t) * t**2 * y3 + t**3 * y4
    return int(x), int(y)

def get_segment_length(
    point_1: MotionVectorPointDict,
    point_2: MotionVectorPointDict
) -> float:
    """
    Gets the length between two points of a motion vector segment
    """
    if point_1.get("control_2", None) is not None and point_2.get("control_1", None) is not None:
        return cubic_bezier_length(
            point_1["anchor"],
            point_1["control_2"],
            point_2["control_1"],
            point_2["anchor"]
        )
    elif point_1.get("control_2", None) is not None and point_2.get("control_1", None) is None:
        return quadratic_bezier_length(
            point_1["anchor"],
            point_1["control_2"],
            point_2["anchor"]
        )
    elif point_1.get("control_2", None) is None and point_2.get("control_1", None) is not None:
        return quadratic_bezier_length(
            point_1["anchor"],
            point_2["control_1"],
            point_2["anchor"]
        )
    else:
        return linear_length(
            point_1["anchor"],
            point_2["anchor"]
        )

def get_segment_point(
    point_1: MotionVectorPointDict,
    point_2: MotionVectorPointDict,
    t: float
) -> Tuple[int, int]:
    """
    Gets a point between two points of a motion vector segment
    """
    if point_1.get("control_2", None) is not None and point_2.get("control_1", None) is not None:
        return cubic_bezier_point(
            point_1["anchor"],
            point_1["control_2"],
            point_2["control_1"],
            point_2["anchor"],
            t
        )
    elif point_1.get("control_2", None) is not None and point_2.get("control_1", None) is None:
        return quadratic_bezier_point(
            point_1["anchor"],
            point_1["control_2"],
            point_2["anchor"],
            t
        )
    elif point_1.get("control_2", None) is None and point_2.get("control_1", None) is not None:
        return quadratic_bezier_point(
            point_1["anchor"],
            point_2["control_1"],
            point_2["anchor"],
            t
        )
    else:
        return linear_point(
            point_1["anchor"],
            point_2["anchor"],
            t
        )

def get_point_along_vector(
    vector: List[MotionVectorPointDict],
    t: float
) -> Tuple[int, int]:
    """
    Given a list of points, calculate the position at time t
    """
    segment_lengths = [ # Assumes linear for ease of calculation
        get_segment_length(vector[i], vector[i+1])
        for i in range(len(vector) - 1)
    ]
    total_segment_length = sum(segment_lengths)
    length_at_t = total_segment_length * t
    running_length = 0.0
    for i, segment_length in enumerate(segment_lengths):
        starting_length = running_length
        running_length += segment_length
        if running_length >= length_at_t:
            # Point lies along this segment
            this_t = (length_at_t - starting_length) / segment_length
            return get_segment_point(vector[i], vector[i+1], this_t)
    return vector[-1]["anchor"]

def get_gaussian_kernel(
    kernel_size: int=199,
    sigma: int=20,
    channels: int=2
) -> Conv2d:
    """
    Creates a gaussian kernel for filtering
    """
    import torch
    import torch.nn as nn
    grid = torch.arange(kernel_size).repeat(kernel_size).view(kernel_size, kernel_size)
    xy_grid = torch.stack([grid, grid.t()], dim=-1).float()
    mean = (kernel_size - 1)/2.0
    variance = sigma**2.0
    kernel = torch.exp(-torch.sum((xy_grid - mean)**2.0, dim=-1) / (2*variance))
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    conv = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        groups=channels,
        bias=False,
    )
    conv.weight.data = kernel
    conv.weight.requires_grad = False
    return conv

def apply_gaussian_kernel(
    tensor: Tensor,
    kernel_size: int=199,
    sigma: int=20,
) -> Tensor:
    """
    Gets then applies a gaussian kernel
    """
    from einops import rearrange
    conv = get_gaussian_kernel(
        channels=tensor.shape[-1],
        kernel_size=kernel_size,
        sigma=sigma
    )
    conv.to(device=tensor.device, dtype=tensor.dtype)
    return rearrange(
        conv(rearrange(tensor, "l h w c -> l c h w")),
        "l c h w -> l h w c"
    )

def optical_flow_conditioning_tensor(
    flow: NDArray,
    gaussian_kernel_size: Optional[int]=199,
    gaussian_sigma: Optional[int]=None,
    device: Optional[Union[str, Device]]=None,
    dtype: Optional[DType]=None,
) -> Tensor:
    """
    Turns an optical flow result from CV into a conditioning tensor
    """
    import torch
    condition = torch.tensor(flow)
    if device is not None:
        condition = condition.to(device=device)
    if dtype is not None:
        condition = condition.to(dtype=dtype)
    if gaussian_sigma is not None and gaussian_kernel_size is not None:
        condition = apply_gaussian_kernel(
            condition,
            kernel_size=gaussian_kernel_size,
            sigma=gaussian_sigma
        )

    f, h, w, c = condition.shape

    return torch.cat([
        torch.zeros((1, h, w, 2), device=device, dtype=dtype),
        condition
    ], dim=0)

def motion_vector_conditioning_tensor(
    width: int,
    height: int,
    frames: int,
    motion_vectors: Optional[List[List[MotionVectorPointDict]]],
    gaussian_kernel_size: int=199,
    gaussian_sigma: int=20,
    device: Optional[Union[str, Device]]=None,
    dtype: Optional[DType]=None,
) -> Tensor:
    """
    Creates tensors out of a list of motion vectors.
    Motion vectors are themselves a list of dictionaries containing anchor
    points and optional control points for bezier curves.
    """
    import torch
    condition = torch.zeros(frames-1, height, width, 2) # f h w c
    if device is not None:
        condition = condition.to(device=device)
    if dtype is not None:
        condition = condition.to(dtype=dtype)
    if not motion_vectors:
        motion_vectors = []
    for motion_vector in motion_vectors:
        all_points = [motion_vector[0]["anchor"]]
        for i in range(frames - 3):
            t = (i + 1) / (frames - 1)
            all_points.append(get_point_along_vector(motion_vector, t))
        all_points.append(motion_vector[-1]["anchor"])
        for i in range(frames - 2):
            segment_start_x, segment_start_y = all_points[i]
            segment_end_x, segment_end_y = all_points[i+1]
            segment_start_x = int(segment_start_x)
            segment_start_y = int(segment_start_y)
            segment_end_x = int(segment_end_x)
            segment_end_y = int(segment_end_y)
            if (
                segment_start_y < 0 or
                segment_start_y >= height or
                segment_start_x < 0 or
                segment_start_x >= width
            ):
                continue
            condition[i][segment_start_y][segment_start_x][0] = segment_end_x - segment_start_x
            condition[i][segment_start_y][segment_start_x][1] = segment_end_y - segment_start_y

    return torch.cat([
        torch.zeros(1, height, width, 2, device=condition.device, dtype=condition.dtype),
        apply_gaussian_kernel(
            condition,
            kernel_size=gaussian_kernel_size,
            sigma=gaussian_sigma
        ),
    ], dim=0)

def flow_condition_to_image_sequence(flow: Tensor) -> List[Image]:
    """
    Converts the motion vector to an image sequence for debugging
    """
    from enfugue.diffusion.util.vision_util import ComputerVision
    return [
        ComputerVision.flow_to_image(frame.float().cpu().numpy())
        for frame in flow
    ]
