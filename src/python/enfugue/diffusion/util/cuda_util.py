from __future__ import annotations

import os

__all__ = ["get_cudnn_lib_dir"]

def get_cudnn_lib_dir() -> str:
    """
    Gets the CUDNN directory
    """
    import nvidia
    import nvidia.cudnn
    cudnn_dir = os.path.dirname(nvidia.cudnn.__file__)
    cudnn_lib_dir = os.path.join(cudnn_dir, "lib")
    if os.path.exists(cudnn_lib_dir):
        return cudnn_lib_dir
    raise IOError("Couldn't find CUDNN directory.")
