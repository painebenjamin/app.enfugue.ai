from typing import TypedDict
from enfugue.util.gputil import getGPUs  # type: ignore[attr-defined]

__all__ = ["GPUMemoryStatusDict", "GPUStatusDict", "get_gpu_status"]


class GPUMemoryStatusDict(TypedDict):
    """
    The memory status dictionary.
    """

    free: int
    total: int
    used: int
    util: float


class GPUStatusDict(TypedDict):
    """
    The GPU status dictionary.
    """

    driver: str
    name: str
    load: float
    temp: float
    memory: GPUMemoryStatusDict


def get_gpu_status() -> GPUStatusDict:
    """
    Gets current GPU status.
    """

    primary_gpu = getGPUs()[0]

    return {
        "driver": primary_gpu.driver,
        "name": primary_gpu.name,
        "load": primary_gpu.load,
        "temp": primary_gpu.temperature,
        "memory": {
            "free": primary_gpu.memoryFree,
            "total": primary_gpu.memoryTotal,
            "used": primary_gpu.memoryUsed,
            "util": primary_gpu.memoryUtil,
        },
    }
