from __future__ import annotations

import os
import json
import platform

from typing import Iterator, Dict, Any, TypedDict, Optional, Union, List

from subprocess import Popen, PIPE
from distutils import spawn

from enfugue.util.log import logger

__all__ = ["get_gpu_status", "GPUMemoryStatusDict", "GPUStatusDict", "GPU"]


def get_gpu_status() -> Optional[GPUStatusDict]:
    """
    Gets current GPU status.
    """
    gpus = GPU.get_gpus()
    if not gpus:
        return None
    primary_gpu = gpus[0]
    return {
        "driver": primary_gpu.driver,
        "name": primary_gpu.name,
        "load": primary_gpu.load,
        "temp": primary_gpu.temp,
        "memory": {
            "free": int(primary_gpu.memory_free),
            "total": int(primary_gpu.memory_total),
            "used": int(primary_gpu.memory_used),
            "util": primary_gpu.memory_util,
        },
    }


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


class GPU:
    """
    This class holds details about the GPU as returned by the appropriate subprocess.
    """

    def __init__(
        self,
        id: str,
        uuid: str,
        load: Union[int, float],
        memory_total: Union[int, float],
        memory_used: Union[int, float],
        temp: Union[int, float],
        driver: str,
        name: str,
    ) -> None:
        self.id = id
        self.uuid = uuid
        self.load = load
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.temp = temp
        self.driver = driver
        self.name = name

    @property
    def memory_util(self) -> float:
        """
        Calculate utilization
        """
        return float(self.memory_used) / float(self.memory_total)

    @property
    def memory_free(self) -> float:
        """
        Calculate free bytes
        """
        return self.memory_total - self.memory_used

    @staticmethod
    def get_process_kwargs() -> Dict[str, Any]:
        """
        Gets keyword arguments to pass into the Popen call.
        """
        process_kwargs: Dict[str, Any] = {"stdout": PIPE, "stderr": PIPE}
        if platform.system() == "Windows":
            from subprocess import CREATE_NO_WINDOW  # type: ignore

            process_kwargs["creationflags"] = CREATE_NO_WINDOW  # type: ignore
        return process_kwargs

    @staticmethod
    def get_nvidia_gpus(executable: str) -> Iterator[GPU]:
        """
        Executes `nvidia-smi` and parses output.
        """
        stderr: str = ""
        try:
            p = Popen(
                [
                    executable,
                    "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                **GPU.get_process_kwargs(),
            )
            stdout, stderr = p.communicate()
            output = stdout.decode("UTF-8")
            lines = [line for line in [line.strip() for line in output.split(os.linesep)] if line]
            for line in lines:
                (id, uuid, load, memory_total, memory_used, _, driver, name, _, _, _, temp) = line.split(",")

                yield GPU(
                    id=id,
                    uuid=uuid.strip(),
                    memory_total=float(memory_total),
                    memory_used=float(memory_used),
                    load=float(load) / 100.0,
                    temp=float(temp),
                    driver=driver.strip(),
                    name=name.strip(),
                )
        except Exception as ex:
            logger.error(f"Couldn't execute nvidia-smi (binary `{executable}`): {ex}\n{stderr}")

    @staticmethod
    def get_amd_gpus(executable: str) -> Iterator[GPU]:
        """
        Executes `rocm-smi` and parses output.
        """
        stderr: str = ""
        try:
            p = Popen(
                [
                    executable,
                    "--alldevices",
                    "-tu",
                    "--showproductname",
                    "--showdriverversion",
                    "--showuniqueid",
                    "--showmeminfo",
                    "vram",
                    "--json",
                ],
                **GPU.get_process_kwargs(),
            )
            stdout, stderr = p.communicate()
            output = stdout.decode("UTF-8")
            result = json.loads(output)
            for key in result:
                if key == "system":
                    continue
                gpu_dict = result[key]
                yield GPU(
                    id=key,
                    uuid=gpu_dict["Unique ID"],
                    memory_total=float(gpu_dict["VRAM Total Memory (B)"]) / 1000000.0,
                    memory_used=float(gpu_dict["VRAM Total Used Memory (B)"]) / 1000000.0,
                    temp=float(gpu_dict["Temperature (Sensor junction) (C)"]),
                    load=float(gpu_dict["GPU use (%)"]) / 100.0,
                    driver=result["system"]["Driver version"],
                    name=gpu_dict["Card series"],
                )

        except Exception as ex:
            logger.error(f"Couldn't execute rocm-smi (binary `{executable}`): {ex}\n{stderr}")

    @staticmethod
    def get_gpus() -> List[GPU]:
        """
        Gets the appropriate binary and executes it
        """
        if platform.system() == "Darwin":
            # MacOS - can't do this yet
            return []

        nvidia_smi = spawn.find_executable("nvidia-smi")
        rocm_smi = spawn.find_executable("rocm-smi")

        if rocm_smi is not None:
            return list(GPU.get_amd_gpus(rocm_smi))
        elif nvidia_smi is None:
            if platform.system() == "Windows":
                nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ["systemdrive"]
            else:
                nvidia_smi = "nvidia-smi"
        return list(GPU.get_nvidia_gpus(nvidia_smi))
