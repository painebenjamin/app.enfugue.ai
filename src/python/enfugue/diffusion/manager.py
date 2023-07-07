from __future__ import annotations

import os
import time
import torch
import random
import datetime
import traceback
import threading

from typing import Type, Union, Any, Optional, List, Tuple, Dict, Literal, Callable, TYPE_CHECKING
from hashlib import md5

from pibble.api.configuration import APIConfiguration
from pibble.api.exceptions import ConfigurationError
from pibble.util.files import dump_json

from enfugue.util import logger, check_download, check_make_directory
from enfugue.diffusion.constants import (
    DEFAULT_MODEL,
    DEFAULT_INPAINTING_MODEL,
    CONTROLNET_CANNY,
    CONTROLNET_MLSD,
    CONTROLNET_HED,
    CONTROLNET_SCRIBBLE,
    CONTROLNET_TILE,
    CONTROLNET_INPAINT,
)

__all__ = ["DiffusionPipelineManager"]

if TYPE_CHECKING:
    from diffusers.models import ControlNetModel
    from enfugue.diffusion.upscale import Upscaler
    from enfugue.diffusion.pipeline import EnfugueStableDiffusionPipeline
    from enfugue.diffusion.edge.detect import EdgeDetector


class KeepaliveThread(threading.Thread):
    """
    Calls the keepalive function every <n> seconds.
    """

    INTERVAL = 0.5
    KEEPALIVE_INTERVAL = 10

    def __init__(self, manager: DiffusionPipelineManager) -> None:
        super(KeepaliveThread, self).__init__()
        self.manager = manager
        self.stop_event = threading.Event()

    @property
    def stopped(self) -> bool:
        """
        Returns true IFF the stop event is set.
        """
        return self.stop_event.is_set()

    def stop(self) -> None:
        """
        Stops the thread.
        """
        self.stop_event.set()

    def run(self) -> None:
        """
        The threading run loop.
        """
        last_keepalive = datetime.datetime.now()
        while not self.stopped:
            time.sleep(self.INTERVAL)
            now = datetime.datetime.now()
            if (now - last_keepalive).total_seconds() > self.KEEPALIVE_INTERVAL:
                self.manager.keepalive_callback()
                last_keepalive = now


class DiffusionPipelineManager:
    TENSORRT_STAGES = [
        "unet"
    ]  # TODO: Get others to work with multidiff (clip works but isnt worth it right now)
    TENSORRT_ALWAYS_USE_CONTROLLED_UNET = False  # TODO: Figure out if this is possible

    PIPELINE_CLASS = "enfugue.diffusion.pipeline.EnfugueStableDiffusionPipeline"
    TRT_PIPELINE_CLASS = "enfugue.diffusion.rt.pipeline.EnfugueTensorRTStableDiffusionPipeline"

    DEFAULT_CHUNK = 64
    DEFAULT_SIZE = 512

    _keepalive_thread: KeepaliveThread

    def __init__(self, configuration: Optional[APIConfiguration] = None) -> None:
        self.configuration = APIConfiguration()
        if configuration:
            self.configuration = configuration

    @property
    def safe(self) -> bool:
        """
        Returns true if safety checking should be enabled.
        """
        if not hasattr(self, "_safe"):
            self._safe = self.configuration.get("enfugue.safe", True)
        return self._safe

    @safe.setter
    def safe(self, val: bool) -> None:
        """
        Sets a new value for safety checking. Destroys the pipeline.
        """
        if val != getattr(self, "_safe", None):
            self._safe = val
            del self.pipeline

    @property
    def device(self) -> torch.device:
        """
        Gets the device that will be executed on
        """
        if not hasattr(self, "_device"):
            from enfugue.diffusion.util import get_optimal_device

            self._device = get_optimal_device()
        return self._device

    @property
    def seed(self) -> int:
        """
        Gets the seed. If there is none, creates a random one once.
        """
        if not hasattr(self, "_seed"):
            self._seed = self.configuration.get("enfugue.seed", random.randint(0, 2**63 - 1))
        return self._seed

    @seed.setter
    def seed(self, new_seed: int) -> None:
        """
        Re-seeds the pipeline. This deletes the generator so it gets re-initialized.
        """
        self._seed = new_seed
        del self.generator

    @property
    def keepalive_callback(self) -> Callable[[], None]:
        """
        A callback function to call during long operations.
        """
        if not hasattr(self, "_keepalive_callback"):
            return lambda: None
        return self._keepalive_callback

    @keepalive_callback.setter
    def keepalive_callback(self, new_callback: Callable[[], None]) -> None:
        """
        Sets the callback
        """
        self._keepalive_callback = new_callback

    @keepalive_callback.deleter
    def keepalive_callback(self) -> None:
        """
        Removes the callback.
        """
        if hasattr(self, "_keepalive_callback"):
            del self._keepalive_callback

    def start_keepalive(self) -> None:
        """
        Starts a thread which will call the keepalive callback every <n> seconds.
        """
        if not hasattr(self, "_keepalive_thread") or not self._keepalive_thread.is_alive():
            keepalive_callback = self.keepalive_callback
            self._keepalive_thread = KeepaliveThread(self)
            self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        """
        Stops the thread which calls the keepalive.
        """
        if hasattr(self, "_keepalive_thread"):
            if self._keepalive_thread.is_alive():
                self._keepalive_thread.stop()
                self._keepalive_thread.join()
            del self._keepalive_thread

    @property
    def generator(self) -> torch.Generator:
        """
        Creates the generator once, otherwise returns it.
        """
        if not hasattr(self, "_generator"):
            try:
                self._generator = torch.Generator(device=self.device)
            except RuntimeError:
                # Unsupported device, go to CPU
                self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        return self._generator

    @generator.deleter
    def generator(self) -> None:
        """
        Removes an existing generator.
        """
        if hasattr(self, "_generator"):
            delattr(self, "_generator")

    @property
    def size(self) -> int:
        """
        Gets the base engine size in pixels when chunking (default always.)
        """
        if not hasattr(self, "_size"):
            self._size = int(
                self.configuration.get("enfugue.size", DiffusionPipelineManager.DEFAULT_SIZE)
            )
        return self._size

    @size.setter
    def size(self, new_size: int) -> None:
        """
        Sets the base engine size in pixels.
        """
        if hasattr(self, "_size") and self._size != new_size:
            del self.pipeline
        self._size = new_size

    @property
    def chunking_size(self) -> int:
        """
        Gets the chunking size in pixels.
        """
        if not hasattr(self, "_chunking_size"):
            self._chunking_size = int(
                self.configuration.get("enfugue.chunk.size", DiffusionPipelineManager.DEFAULT_CHUNK)
            )
        return self._chunking_size

    @chunking_size.setter
    def chunking_size(self, new_chunking_size: int) -> None:
        """
        Sets the new chunking size. This doesn't require a restart.
        """
        self._chunking_size = new_chunking_size

    @property
    def chunking_blur(self) -> int:
        """
        Gets the chunking blur in pixels.
        """
        if not hasattr(self, "_chunking_blur"):
            self._chunking_blur = int(
                self.configuration.get("enfugue.chunk.blur", DiffusionPipelineManager.DEFAULT_CHUNK)
            )
        return self._chunking_blur

    @chunking_blur.setter
    def chunking_blur(self, new_chunking_blur: int) -> None:
        """
        Sets the new chunking blur. This doesn't require a restart.
        """
        self._chunking_blur = new_chunking_blur

    @property
    def engine_root(self) -> str:
        """
        Gets the root of the engine.
        """
        path = self.configuration.get("enfugue.engine.root", "~/.cache/enfugue")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def engine_cache_dir(self) -> str:
        """
        Gets the cache for diffusers-downloaded configuration files, base models, etc.
        """
        path = self.configuration.get("enfugue.engine.cache", "~/.cache/enfugue/cache")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def engine_checkpoints_dir(self) -> str:
        """
        Gets where checkpoints are downloaded in.
        """
        path = self.configuration.get("enfugue.engine.checkpoint", "~/.cache/enfugue/checkpoint")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def engine_other_dir(self) -> str:
        """
        Gets where any other weights are download in
        """
        path = self.configuration.get("enfugue.engine.other", "~/.cache/enfugue/other")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def engine_lora_dir(self) -> str:
        """
        Gets where lora are downloaded in.
        """
        path = self.configuration.get("enfugue.engine.lora", "~/.cache/enfugue/lora")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def engine_inversion_dir(self) -> str:
        """
        Gets where inversion are downloaded to.
        """
        path = self.configuration.get("enfugue.engine.inversion", "~/.cache/enfugue/inversion")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def engine_tensorrt_dir(self) -> str:
        """
        Gets where TensorRT engines are downloaded to.
        """
        path = self.configuration.get("enfugue.engine.tensorrt", "~/.cache/enfugue/tensorrt")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def model_tensorrt_dir(self) -> str:
        """
        Gets where tensorrt engines will be built per model.
        """
        path = os.path.join(self.engine_tensorrt_dir, self.model_name)
        check_make_directory(path)
        return path
    
    @property
    def refiner_tensorrt_dir(self) -> str:
        """
        Gets where tensorrt engines will be built per refiner.
        """
        path = os.path.join(self.engine_tensorrt_dir, self.refiner_name)
        check_make_directory(path)
        return path

    @staticmethod
    def get_tensorrt_clip_key(
        size: int,
        lora: List[Tuple[str, float]],
        lycoris: List[Tuple[str, float]],
        inversion: List[str],
        **kwargs: Any
    ) -> str:
        """
        Uses hashlib to generate the unique key for the CLIP engine.
        CLIP must be rebuilt for each:
            1. Model
            2. Dimension
            3. LoRA
            4. LyCORIS
            5. Textual Inversion
        """
        return md5(
            "-".join(
                [
                    str(size),
                    ":".join(
                        "=".join([str(part) for part in lora_weight])
                        for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                    ),
                    ":".join(
                        "=".join([str(part) for part in lycoris_weight])
                        for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                    ),
                    ":".join(sorted(inversion)),
                ]
            ).encode("utf-8")
        ).hexdigest()

    @property
    def model_tensorrt_clip_key(self) -> str:
        """
        Gets the CLIP key for the current configuration.
        """
        return DiffusionPipelineManager.get_tensorrt_clip_key(
            size=self.size,
            lora=self.lora_names_weights,
            lycoris=self.lycoris_names_weights,
            inversion=self.inversion_names,
        )

    @property
    def model_tensorrt_clip_dir(self) -> str:
        """
        Gets where the tensorrt CLIP engine will be stored.
        """
        path = os.path.join(self.model_tensorrt_dir, "clip", self.model_tensorrt_clip_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path
    
    @property
    def refiner_tensorrt_clip_dir(self) -> str:
        """
        Gets where the tensorrt CLIP engine will be stored.
        """
        path = os.path.join(self.refiner_tensorrt_dir, "clip", self.model_tensorrt_clip_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path

    @staticmethod
    def get_tensorrt_unet_key(
        size: int,
        lora: List[Tuple[str, float]],
        lycoris: List[Tuple[str, float]],
        inversion: List[str],
        **kwargs: Any,
    ) -> str:
        """
        Uses hashlib to generate the unique key for the UNET engine.
        UNET must be rebuilt for each:
            1. Model
            2. Dimension
            3. LoRA
            4. LyCORIS
            5. Textual Inversion
        """
        return md5(
            "-".join(
                [
                    str(size),
                    ":".join(
                        "=".join([str(part) for part in lora_weight])
                        for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                    ),
                    ":".join(
                        "=".join([str(part) for part in lycoris_weight])
                        for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                    ),
                    ":".join(sorted(inversion)),
                ]
            ).encode("utf-8")
        ).hexdigest()

    @property
    def model_tensorrt_unet_key(self) -> str:
        """
        Gets the UNET key for the current configuration.
        """
        return DiffusionPipelineManager.get_tensorrt_unet_key(
            size=self.size,
            lora=self.lora_names_weights,
            lycoris=self.lycoris_names_weights,
            inversion=self.inversion_names,
        )

    @property
    def model_tensorrt_unet_dir(self) -> str:
        """
        Gets where the tensorrt UNET engine will be stored.
        """
        path = os.path.join(self.model_tensorrt_dir, "unet", self.model_tensorrt_unet_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path
    
    @property
    def refiner_tensorrt_unet_dir(self) -> str:
        """
        Gets where the tensorrt UNET engine will be stored for the refiner.
        """
        path = os.path.join(self.refiner_tensorrt_dir, "unet", self.model_tensorrt_unet_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path

    @staticmethod
    def get_tensorrt_controlled_unet_key(
        size: int,
        lora: List[Tuple[str, float]],
        lycoris: List[Tuple[str, float]],
        inversion: List[str],
        **kwargs: Any,
    ) -> str:
        """
        Uses hashlib to generate the unique key for the UNET engine with controlnet blocks.
        ControlledUNET must be rebuilt for each:
            1. Model
            2. Dimension
            3. LoRA
            4. LyCORIS
            5. Textual Inversion
        """
        return md5(
            "-".join(
                [
                    str(size),
                    ":".join(
                        "=".join([str(part) for part in lora_weight])
                        for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
                    ),
                    ":".join(
                        "=".join([str(part) for part in lycoris_weight])
                        for lycoris_weight in sorted(lycoris, key=lambda lycoris_part: lycoris_part[0])
                    ),
                    ":".join(sorted(inversion)),
                ]
            ).encode("utf-8")
        ).hexdigest()

    @property
    def model_tensorrt_controlled_unet_key(self) -> str:
        """
        Gets the UNET key for the current configuration.
        """
        return DiffusionPipelineManager.get_tensorrt_controlled_unet_key(
            size=self.size,
            lora=self.lora_names_weights,
            lycoris=self.lycoris_names_weights,
            inversion=self.inversion_names,
        )

    @property
    def model_tensorrt_controlled_unet_dir(self) -> str:
        """
        Gets where the tensorrt Controlled UNet engine will be stored.
        """
        path = os.path.join(
            self.model_tensorrt_dir, "controlledunet", self.model_tensorrt_controlled_unet_key
        )
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path
    
    @property
    def refiner_tensorrt_controlled_unet_dir(self) -> str:
        """
        Gets where the tensorrt Controlled UNet engine will be stored for the refiner.
        TODO: determine if this should exist.
        """
        path = os.path.join(
            self.refiner_tensorrt_dir, "controlledunet", self.model_tensorrt_controlled_unet_key
        )
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path

    @staticmethod
    def get_tensorrt_controlnet_key(size: int, controlnet: str, **kwargs: Any) -> str:
        """
        Uses hashlib to generate the unique key for the controlnet engine.
        controlnet must be rebuilt for each:
            1. Model
            2. Dimension
            3. ControlNet
        """
        return md5("-".join([str(size), controlnet]).encode("utf-8")).hexdigest()

    @property
    def model_tensorrt_controlnet_key(self) -> str:
        """
        Gets the UNET key for the current configuration.
        """
        return DiffusionPipelineManager.get_tensorrt_controlnet_key(
            size=self.size,
            controlnet=""
            if self.controlnet is None
            else getattr(self.controlnet.config, "_name_or_path", ""),
        )

    @property
    def model_tensorrt_controlnet_dir(self) -> str:
        """
        Gets where the tensorrt controlnet engine will be stored.
        """
        path = os.path.join(
            self.model_tensorrt_dir, "controlnet", self.model_tensorrt_controlnet_key
        )
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path
    
    @property
    def refiner_tensorrt_controlnet_dir(self) -> str:
        """
        Gets where the tensorrt controlnet engine will be stored for the refiner.
        """
        path = os.path.join(
            self.refiner_tensorrt_dir, "controlnet", self.model_tensorrt_controlnet_key
        )
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path

    @staticmethod
    def get_tensorrt_vae_key(size: int, **kwargs: Any) -> str:
        """
        Uses hashlib to generate the unique key for the VAE engine.
        VAE must be rebuilt for each:
            1. Model
            2. Dimension
        """
        return md5(str(size).encode("utf-8")).hexdigest()

    @property
    def model_tensorrt_vae_key(self) -> str:
        """
        Gets the UNET key for the current configuration.
        """
        return DiffusionPipelineManager.get_tensorrt_vae_key(size=self.size)

    @property
    def model_tensorrt_vae_dir(self) -> str:
        """
        Gets where the tensorrt VAE engine will be stored.
        """
        path = os.path.join(self.model_tensorrt_dir, "vae", self.model_tensorrt_vae_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path
    
    @property
    def refiner_tensorrt_vae_dir(self) -> str:
        """
        Gets where the tensorrt VAE engine will be stored for the refiner.
        """
        path = os.path.join(self.refiner_tensorrt_dir, "vae", self.model_tensorrt_vae_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path

    @property
    def tensorrt_is_supported(self) -> bool:
        """
        Tries to import tensorrt to see if it's supported.
        """
        if not hasattr(self, "_tensorrt_is_supported"):
            try:
                import tensorrt

                tensorrt.__version__  # quiet importchecker
                self._tensorrt_is_supported = True
            except Exception as ex:
                logger.info("TensorRT is disabled.")
                logger.debug("{0}: {1}".format(type(ex).__name__, ex))
                self._tensorrt_is_supported = False
        return self._tensorrt_is_supported

    @property
    def tensorrt_is_enabled(self) -> bool:
        """
        By default this is always enabled. This is independent from supported/ready.
        """
        if not hasattr(self, "_tensorrt_enabled"):
            self._tensorrt_enabled = True
        return self._tensorrt_enabled

    @tensorrt_is_enabled.setter
    def tensorrt_is_enabled(self, new_enabled: bool) -> None:
        """
        Disables or enables TensorRT.
        """
        if new_enabled != self.tensorrt_is_enabled and self.tensorrt_is_ready:
            del self.pipeline
        self._tensorrt_enabled = new_enabled

    @property
    def tensorrt_is_ready(self) -> bool:
        """
        Checks to determine if Tensor RT is ready based on the existence of engines.
        """
        if not self.tensorrt_is_supported:
            return False
        from enfugue.diffusion.rt.engine import Engine

        trt_ready = True
        if "vae" in self.TENSORRT_STAGES:
            trt_ready = trt_ready and os.path.exists(
                Engine.get_engine_path(self.model_tensorrt_vae_dir)
            )
        if "clip" in self.TENSORRT_STAGES:
            trt_ready = trt_ready and os.path.exists(
                Engine.get_engine_path(self.model_tensorrt_clip_dir)
            )
        if self.controlnet is not None or self.TENSORRT_ALWAYS_USE_CONTROLLED_UNET:
            if "unet" in self.TENSORRT_STAGES:
                trt_ready = trt_ready and os.path.exists(
                    Engine.get_engine_path(self.model_tensorrt_controlled_unet_dir)
                )
            if "controlnet" in self.TENSORRT_STAGES:
                trt_ready = trt_ready and os.path.exists(
                    Engine.get_engine_path(self.model_tensorrt_controlnet_dir)
                )
        elif "unet" in self.TENSORRT_STAGES:
            trt_ready = trt_ready and os.path.exists(
                Engine.get_engine_path(self.model_tensorrt_unet_dir)
            )
        return trt_ready

    @property
    def build_tensorrt(self) -> bool:
        """
        Checks to see if TensorRT should be built based on configuration.
        """
        if not hasattr(self, "_build_tensorrt"):
            if not self.tensorrt_is_supported:
                self._build_tensorrt = False
            else:
                self._build_tensorrt = self.configuration.get("enfugue.tensorrt", False)
        return self._build_tensorrt

    @build_tensorrt.setter
    def build_tensorrt(self, new_build: bool) -> None:
        """
        Changes whether or not TensorRT engines should be built when absent
        """
        self._build_tensorrt = new_build
        if not self.tensorrt_is_ready and self.tensorrt_is_supported:
            del self.pipeline  # Prepare for build

    @property
    def use_tensorrt(self) -> bool:
        """
        Gets the ultimate decision on whether the tensorrt pipeline should be used.
        """
        return (self.tensorrt_is_ready or self.build_tensorrt) and self.tensorrt_is_enabled

    @property
    def refiner_switch_mode(self) -> Optional[Literal["offload", "delete"]]:
        """
        Defines how to switch to refiners.
        """
        if not hasattr(self, "_refiner_switch_mode"):
            self._refiner_switch_mode = self.configuration.get("enfugue.refiner.switch", "offload")
        return self._refiner_switch_mode

    @refiner_switch_mode.setter
    def refiner_switch_mode(self, mode: Optional[Literal["offload", "delete"]]) -> None:
        """
        Changes how refiners get switched.
        """
        self._refiner_switch_mode = mode

    @property
    def refiner_strength(self) -> float:
        """
        Gets the denoising strength of the refiner
        """
        if not hasattr(self, "_refiner_strength"):
            self._refiner_strength = self.configuration.get("enfugue.refiner.strength", 0.3)
        return self._refiner_strength
    
    @refiner_strength.setter
    def refiner_strength(self, new_strength: float) -> None:
        """
        Sets the denoising strength of the refiner
        """
        self._refiner_strength = new_strength
    
    @property
    def refiner_guidance_scale(self) -> float:
        """
        Gets the guidance_ cale of the refiner
        """
        if not hasattr(self, "_refiner_guidance_scale"):
            self._refiner_guidance_scale = self.configuration.get("enfugue.refiner.guidance_scale", 5.0)
        return self._refiner_guidance_scale
    
    @refiner_guidance_scale.setter
    def refiner_guidance_scale(self, new_guidance_scale: float) -> None:
        """
        Sets the guidance scale of the refiner
        """
        self._refiner_guidance_scale = new_guidance_scale
    
    @property
    def aesthetic_score(self) -> float:
        """
        Gets the aesthetic score for the refiner
        """
        if not hasattr(self, "_aesthetic_score"):
            self._aesthetic_score = self.configuration.get("enfugue.refiner.aesthetic_score", 6.0)
        return self._aesthetic_score
    
    @aesthetic_score.setter
    def aesthetic_score(self, new_aesthetic_score: float) -> None:
        """
        Sets the aesthetic score for the refiner
        """
        self._aesthetic_score = new_aesthetic_score
    
    @property
    def negative_aesthetic_score(self) -> float:
        """
        Gets the negative aesthetic score for the refiner
        """
        if not hasattr(self, "_negative_aesthetic_score"):
            self._negative_aesthetic_score = self.configuration.get("enfugue.refiner.negative_aesthetic_score", 2.5)
        return self._negative_aesthetic_score
    
    @negative_aesthetic_score.setter
    def negative_aesthetic_score(self, new_negative_aesthetic_score: float) -> None:
        """
        Sets the negative aesthetic score for the refiner
        """
        self._negative_aesthetic_score = new_negative_aesthetic_score

    @property
    def pipeline_class(self) -> Type:
        """
        Gets the pipeline class to use.
        """
        if self.use_tensorrt:
            from enfugue.diffusion.rt.pipeline import EnfugueTensorRTStableDiffusionPipeline

            return EnfugueTensorRTStableDiffusionPipeline
        else:
            from enfugue.diffusion.pipeline import EnfugueStableDiffusionPipeline

            return EnfugueStableDiffusionPipeline

    @property
    def model(self) -> str:
        """
        Gets the configured model.
        """
        if not hasattr(self, "_model"):
            self._model = self.configuration.get("enfugue.model", DEFAULT_MODEL)
        return self._model

    @model.setter
    def model(self, new_model: Optional[str]) -> None:
        """
        Sets a new model. Destroys the pipeline.
        """
        if new_model is None:
            new_model = self.configuration.get("enfugue.model", DEFAULT_MODEL)
        if new_model.startswith("http"):
            new_model = self.check_download_checkpoint(new_model)
        elif not os.path.isabs(new_model):
            new_model = os.path.join(self.engine_checkpoints_dir, new_model)
        new_model_name, _ = os.path.splitext(os.path.basename(new_model))
        if self.model_name != new_model_name:
            del self.pipeline
        self._model = new_model

    @property
    def model_name(self) -> str:
        """
        Gets just the basename of the model
        """
        return os.path.splitext(os.path.basename(self.model))[0]

    @property
    def xl(self) -> bool:
        """
        Returns true if this is an XL model (based on filename)
        """
        return "xl" in self.model_name.lower()
    
    @property
    def refiner(self) -> Optional[str]:
        """
        Gets the configured refiner.
        """
        if not hasattr(self, "_refiner"):
            self._refiner = self.configuration.get("enfugue.refiner", None)
        return self._refiner

    @refiner.setter
    def refiner(self, new_refiner: Optional[str]) -> None:
        """
        Sets a new refiner. Destroys the refiner pipelline.
        """
        if new_refiner is None:
            self.refiner = None
            return
        if new_refiner.startswith("http"):
            new_refiner = self.check_download_checkpoint(new_refiner)
        elif not os.path.isabs(new_refiner):
            new_refiner = os.path.join(self.engine_checkpoints_dir, new_refiner)
        new_refiner_name, _ = os.path.splitext(os.path.basename(new_refiner))
        if self.refiner_name != new_refiner_name:
            del self.refiner_pipeline
        self._refiner = new_refiner

    @property
    def refiner_name(self) -> Optional[str]:
        """
        Gets just the basename of the refiner
        """
        if self.refiner is None:
            return None
        return os.path.splitext(os.path.basename(self.refiner))[0]

    @property
    def dtype(self) -> torch.dtype:
        """
        Gets the default or configured torch data type
        """
        if not hasattr(self, "_torch_dtype"):
            if self.device.type == "cpu":
                logger.debug("Inferencing on CPU, using BFloat")
                self._torch_dtype = torch.bfloat16
            else:
                configuration_dtype = self.configuration.get("enfugue.dtype", "float16")
                if configuration_dtype == "float16" or configuration_dtype == "half":
                    self._torch_dtype = torch.half
                elif configuration_dtype == "float32" or configuration_dtype == "float":
                    self._torch_dtype = torch.float
                else:
                    raise ConfigurationError(
                        "dtype incorrectly configured, use 'float16/half' or 'float32/float'"
                    )
        return self._torch_dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]) -> None:
        """
        Sets the torch dtype.
        Deletes the pipeline if it's different from the previous one.
        """
        if self.device.type == "cpu":
            raise ValueError("CPU-based diffusion can only use bfloat")
        if new_dtype == "float16" or new_dtype == "half":
            new_dtype = torch.half
        elif new_dtype == "float32" or new_dtype == "float":
            new_dtype = torch.float
        else:
            raise ConfigurationError(
                "dtype incorrectly configured, use 'float16/half' or 'float32/float'"
            )

        if getattr(self, "_torch_dtype", new_dtype) != new_dtype:
            del self.pipeline

        self._torch_dtype = new_dtype

    @property
    def lora(self) -> List[Tuple[str, float]]:
        """
        Get LoRA added to the text encoder and UNet.
        """
        return getattr(self, "_lora", [])

    @lora.setter
    def lora(
        self, new_lora: Optional[Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]]
    ) -> None:
        """
        Sets new LoRA. Destroys the pipeline.
        """
        if new_lora is None:
            if hasattr(self, "_lora") and len(self._lora) > 0:
                del self.pipeline
            self._lora: List[Tuple[str, float]] = []
            return

        lora: List[Tuple[str, float]] = []
        if isinstance(new_lora, list):
            for this_lora in new_lora:
                if isinstance(this_lora, list):
                    lora.append(tuple(this_lora))  # type: ignore[unreachable]
                elif isinstance(this_lora, tuple):
                    lora.append(this_lora)
                else:
                    lora.append((this_lora, 1))
        elif isinstance(new_lora, tuple):
            lora = [new_lora]
        else:
            lora = [(new_lora, 1)]

        if getattr(self, "_lora", []) != lora:
            del self.pipeline
            self._lora = lora

    @property
    def lora_names_weights(self) -> List[Tuple[str, float]]:
        """
        Gets the basenames of any LoRA present.
        """
        return [(os.path.splitext(os.path.basename(lora))[0], weight) for lora, weight in self.lora]

    @property
    def lycoris(self) -> List[Tuple[str, float]]:
        """
        Get lycoris added to the text encoder and UNet.
        """
        return getattr(self, "_lycoris", [])

    @lycoris.setter
    def lycoris(
        self, new_lycoris: Optional[Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]]
    ) -> None:
        """
        Sets new lycoris. Destroys the pipeline.
        """
        if new_lycoris is None:
            if hasattr(self, "_lycoris") and len(self._lycoris) > 0:
                del self.pipeline
            self._lycoris: List[Tuple[str, float]] = []
            return

        lycoris: List[Tuple[str, float]] = []
        if isinstance(new_lycoris, list):
            for this_lycoris in new_lycoris:
                if isinstance(this_lycoris, list):
                    lycoris.append(tuple(this_lycoris))  # type: ignore[unreachable]
                elif isinstance(this_lycoris, tuple):
                    lycoris.append(this_lycoris)
                else:
                    lycoris.append((this_lycoris, 1))
        elif isinstance(new_lycoris, tuple):
            lycoris = [new_lycoris]
        else:
            lycoris = [(new_lycoris, 1)]

        if getattr(self, "_lycoris", []) != lycoris:
            del self.pipeline
            self._lycoris = lycoris

    @property
    def lycoris_names_weights(self) -> List[Tuple[str, float]]:
        """
        Gets the basenames of any lycoris present.
        """
        return [(os.path.splitext(os.path.basename(lycoris))[0], weight) for lycoris, weight in self.lycoris]

    @property
    def inversion(self) -> List[str]:
        """
        Get textual inversion added to the text encoder.
        """
        return getattr(self, "_inversion", [])

    @inversion.setter
    def inversion(self, new_inversion: Optional[Union[str, List[str]]]) -> None:
        """
        Sets new textual inversion. Destroys the pipeline.
        """
        if new_inversion is None:
            if hasattr(self, "_inversion") and len(self._inversion) > 0:
                del self.pipeline
            self._inversion: List[str] = []
            return

        if not isinstance(new_inversion, list):
            new_inversion = [new_inversion]
        if getattr(self, "_inversion", []) != new_inversion:
            del self.pipeline
            self._inversion = new_inversion

    @property
    def inversion_names(self) -> List[str]:
        """
        Gets the basenames of any textual inversions present.
        """
        return [os.path.splitext(os.path.basename(inversion))[0] for inversion in self.inversion]

    @property
    def inpainting(self) -> bool:
        """
        Returns true if the model is an inpainting model.
        """
        return "inpaint" in self.model

    @inpainting.setter
    def inpainting(self, new_inpainting: bool) -> None:
        """
        Sets whether or not we are inpainting.

        We trade efficiency for ease-of-use here; we just keep a model named `-inpainting`
        for any model.
        """
        if self.inpainting != new_inpainting:
            del self.pipeline

            current_checkpoint_path = self.model
            default_checkpoint_name, _ = os.path.splitext(os.path.basename(DEFAULT_MODEL))
            default_inpainting_name, _ = os.path.splitext(
                os.path.basename(DEFAULT_INPAINTING_MODEL)
            )
            checkpoint_name, ext = os.path.splitext(os.path.basename(current_checkpoint_path))

            if default_checkpoint_name == checkpoint_name and new_inpainting:
                logger.info(f"Switching to default inpainting checkpoint")
                self.model = self.check_download_checkpoint(DEFAULT_INPAINTING_MODEL)
            elif default_inpainting_name == checkpoint_name and not new_inpainting:
                logger.info(f"Switching to default model")
                self.model = self.check_download_checkpoint(DEFAULT_MODEL)
            else:
                target_checkpoint_name = checkpoint_name

                if new_inpainting:
                    target_checkpoint_name += "-inpainting"
                else:
                    target_checkpoint_name = target_checkpoint_name[:-11]
                target_checkpoint_path = os.path.join(
                    os.path.dirname(current_checkpoint_path), f"{target_checkpoint_name}{ext}"
                )
                if not os.path.exists(target_checkpoint_path):
                    if not new_inpainting:
                        raise NotImplementedError(
                            "Creating a non-inpainting checkpoint from an inpainting one is not yet supported."
                        )
                    logger.info(f"Creating inpainting checkpoint from {current_checkpoint_path}")
                    self.create_inpainting_checkpoint(
                        current_checkpoint_path, target_checkpoint_path
                    )
                logger.info(
                    "Switching to {0}inpainting checkpoint at {1}".format(
                        "" if new_inpainting else "non-", target_checkpoint_path
                    )
                )
                self.model = target_checkpoint_path

    @property
    def engine_cache_exists(self) -> bool:
        """
        Gets whether or not the diffusers cache exists.
        """
        return os.path.exists(os.path.join(self.model_tensorrt_dir, "model_index.json"))
    
    @property
    def refiner_engine_cache_exists(self) -> bool:
        """
        Gets whether or not the diffusers cache exists.
        """
        return os.path.exists(os.path.join(self.refiner_tensorrt_dir, "model_index.json"))

    def check_create_tensorrt_engine_cache(self) -> None:
        """
        Converts a .ckpt file to the directory structure from diffusers
            This ensures TRT compatibility
        """
        if not self.engine_cache_exists:
            from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
                download_from_original_stable_diffusion_ckpt,
            )

            self.start_keepalive()
            _, ext = os.path.splitext(self.model)
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=self.model,
                scheduler_type="ddim",
                from_safetensors=ext == ".safetensors",
                num_in_channels=9 if "inpaint" in self.model else 4,
            ).to(torch_dtype=self.dtype)
            pipe.save_pretrained(self.model_tensorrt_dir)
            del pipe
            torch.cuda.empty_cache()
            self.stop_keepalive()

    @property
    def pipeline(self) -> EnfugueStableDiffusionPipeline:
        """
        Instantiates the pipeline.
        """
        if not hasattr(self, "_pipeline"):
            if self.model.startswith("http"):
                # Base model, make sure it's downloaded here
                self.model = self.check_download_checkpoint(self.model)

            kwargs = {
                "cache_dir": self.engine_cache_dir,
                "engine_size": self.size,
                "chunking_size": self.chunking_size,
                "requires_safety_checker": self.safe,
                "torch_dtype": self.dtype,
            }
            controlnet = self.controlnet # Load into memory here

            if self.use_tensorrt:
                if "unet" in self.TENSORRT_STAGES:
                    if self.controlnet is None and not self.TENSORRT_ALWAYS_USE_CONTROLLED_UNET:
                        kwargs["unet_engine_dir"] = self.model_tensorrt_unet_dir
                    else:
                        kwargs[
                            "controlled_unet_engine_dir"
                        ] = self.model_tensorrt_controlled_unet_dir
                if "controlnet" in self.TENSORRT_STAGES and self.controlnet is not None:
                    kwargs["controlnet_engine_dir"] = self.model_tensorrt_controlnet_dir
                if "vae" in self.TENSORRT_STAGES:
                    kwargs["vae_engine_dir"] = self.model_tensorrt_vae_dir
                if "clip" in self.TENSORRT_STAGES:
                    kwargs["clip_engine_dir"] = self.model_tensorrt_clip_dir
                if not self.safe:
                    kwargs["safety_checker"] = None
                self.check_create_tensorrt_engine_cache()
                logger.debug(
                    f"Initializing pipeline from diffusers cache directory at {self.model_tensorrt_dir}. Arguments are {kwargs}"
                )
                pipeline = self.pipeline_class.from_pretrained(
                    self.model_tensorrt_dir,
                    controlnet=controlnet,
                    **kwargs
                )
            elif self.engine_cache_exists:
                if not self.safe:
                    kwargs["safety_checker"] = None
                logger.debug(
                    f"Initializing pipeline from diffusers cache directory at {self.model_tensorrt_dir}. Arguments are {kwargs}"
                )
                pipeline = self.pipeline_class.from_pretrained(
                    self.model_tensorrt_dir,
                    controlnet=controlnet,
                    **kwargs
                )
            else:
                kwargs["load_safety_checker"] = self.safe
                logger.debug(
                    f"Initializing pipeline from checkpoint at {self.model}. Arguments are {kwargs}"
                )
                pipeline = self.pipeline_class.from_ckpt(
                    self.model,
                    num_in_channels=9 if self.inpainting else 4,
                    controlnet=controlnet,
                    **kwargs
                )
            if not self.tensorrt_is_ready:
                for lora, weight in self.lora:
                    logger.debug(f"Adding LoRA {lora} to pipeline")
                    pipeline.load_lora_weights(lora, multiplier=weight)
                for lycoris, weight in self.lycoris:
                    logger.debug(f"Adding lycoris {lycoris} to pipeline")
                    pipeline.load_lycoris_weights(lycoris, multiplier=weight)
                for inversion in self.inversion:
                    logger.debug(f"Adding textual inversion {inversion} to pipeline")
                    pipeline.load_textual_inversion(inversion)
            self._pipeline = pipeline.to(self.device)
        return self._pipeline

    @pipeline.deleter
    def pipeline(self) -> None:
        """
        Eliminates any instantiated pipeline.
        """
        if hasattr(self, "_pipeline"):
            logger.debug("Deleting pipeline.")
            del self._pipeline
            import torch

            torch.cuda.empty_cache()
        else:
            logger.debug("Pipeline delete called, but no pipeline present. This is not an error.")

    @property
    def refiner_pipeline(self) -> EnfugueStableDiffusionPipeline:
        """
        Instantiates the refiner pipeline.
        """
        if not hasattr(self, "_refiner_pipeline"):
            if self.refiner.startswith("http"):
                # Base refiner, make sure it's downloaded here
                self.refiner = self.check_download_checkpoint(self.refiner)

            kwargs = {
                "cache_dir": self.engine_cache_dir,
                "engine_size": self.size,
                "chunking_size": self.chunking_size,
                "requires_safety_checker": False,
                "torch_dtype": self.dtype,
                "controlnet": None # TODO: investigate
            }

            if self.use_tensorrt:
                if "unet" in self.TENSORRT_STAGES:
                    if self.controlnet is None and not self.TENSORRT_ALWAYS_USE_CONTROLLED_UNET:
                        kwargs["unet_engine_dir"] = self.refiner_tensorrt_unet_dir
                    else:
                        kwargs[
                            "controlled_unet_engine_dir"
                        ] = self.refiner_tensorrt_controlled_unet_dir
                
                """
                if "controlnet" in self.TENSORRT_STAGES and self.controlnet is not None:
                    kwargs["controlnet_engine_dir"] = self.refiner_tensorrt_controlnet_dir
                """

                if "vae" in self.TENSORRT_STAGES:
                    kwargs["vae_engine_dir"] = self.refiner_tensorrt_vae_dir
                if "clip" in self.TENSORRT_STAGES:
                    kwargs["clip_engine_dir"] = self.refiner_tensorrt_clip_dir
                
                self.check_create_tensorrt_refiner_engine_cache()
                logger.debug(
                    f"Initializing refiner pipeline from diffusers cache directory at {self.refiner_tensorrt_dir}. Arguments are {kwargs}"
                )
                refiner_pipeline = self.pipeline_class.from_pretrained(
                    self.refiner_tensorrt_dir,
                    #controlnet=controlnet,
                    safety_checker=None,
                    **kwargs
                )
            elif self.refiner_engine_cache_exists:
                logger.debug(
                    f"Initializing refiner pipeline from diffusers cache directory at {self.refiner_tensorrt_dir}. Arguments are {kwargs}"
                )
                refiner_pipeline = self.pipeline_class.from_pretrained(
                    self.refiner_tensorrt_dir,
                    safety_checker=None,
                    #controlnet=controlnet,
                    **kwargs
                )
            else:
                logger.debug(
                    f"Initializing refiner pipeline from checkpoint at {self.refiner}. Arguments are {kwargs}"
                )
                refiner_pipeline = self.pipeline_class.from_ckpt(
                    self.refiner,
                    num_in_channels=4,
                    load_safety_checker=False,
                    #controlnet=controlnet,
                    **kwargs
                )
            self._refiner_pipeline = refiner_pipeline.to(self.device)
        return self._refiner_pipeline

    @refiner_pipeline.deleter
    def refiner_pipeline(self) -> None:
        """
        Unloads the refiner pipeline if present.
        """
        if hasattr(self, "_refiner_pipeline"):
            logger.debug("Deleting refiner pipeline.")
            del self._refiner_pipeline
            import torch

            torch.cuda.empty_cache()
        else:
            logger.debug("Refiner pipeline delete called, but no refiner pipeline present. This is not an error.")

    def unload_pipeline(self) -> None:
        """
        Calls the pipeline deleter.
        """
        del self.pipeline

    def offload_pipeline(self) -> None:
        """
        Offloads the pipeline to CPU if present.
        """
        if getattr(self, "_pipeline", None) is not None:
            import torch
            self._pipeline = self._pipeline.to("cpu", torch_dtype=torch.float32)
            torch.cuda.empty_cache()

    def reload_pipeline(self) -> None:
        """
        Reloads the pipeline to the device if present.
        """
        if getattr(self, "_pipeline", None) is not None:
            self._pipeline = self._pipeline.to(self.device, torch_dtype=self.dtype)

    def unload_refiner(self) -> None:
        """
        Calls the refiner deleter.
        """
        del self.refiner_pipeline

    def offload_refiner(self) -> None:
        """
        Offloads the pipeline to CPU if present.
        """
        if getattr(self, "_refiner_pipeline", None) is not None:
            import torch
            self._refiner_pipeline = self._refiner_pipeline.to("cpu", torch_dtype=torch.float32)
            torch.cuda.empty_cache()

    def reload_refiner(self) -> None:
        """
        Reloads the pipeline to the device if present.
        """
        if getattr(self, "_refiner_pipeline", None) is not None:
            self._refiner_pipeline = self._refiner_pipeline.to(self.device, torch_dtype=self.dtype)

    @property
    def upscaler(self) -> Upscaler:
        """
        Gets the GAN upscaler
        """
        if not hasattr(self, "_upscaler"):
            from enfugue.diffusion.upscale import Upscaler

            self._upscaler = Upscaler(self.engine_other_dir)
        return self._upscaler

    @upscaler.deleter
    def upscaler(self) -> None:
        """
        Deletes the upscaler to save VRAM.
        """
        if hasattr(self, "_upscaler"):
            logger.debug("Deleting upscaler.")
            del self._upscaler
            import torch

            torch.cuda.empty_cache()

    @property
    def edge_detector(self) -> EdgeDetector:
        """
        Gets the edge detector.
        """
        if not hasattr(self, "_edge_detector"):
            from enfugue.diffusion.edge.detect import EdgeDetector

            self._edge_detector = EdgeDetector(self.engine_other_dir)
        return self._edge_detector

    def unload_upscaler(self) -> None:
        """
        Calls the upscaler deleter.
        """
        del self.upscaler

    def get_controlnet(self, controlnet: Optional[str] = None) -> Optional[ControlNetModel]:
        """
        Loads a controlnet
        """
        if controlnet is None:
            return None
        from diffusers.models import ControlNetModel

        expected_controlnet_location = os.path.join(
            self.engine_cache_dir, controlnet.replace("/", "--")
        )

        if not os.path.exists(expected_controlnet_location):
            logger.info(
                f"Controlnet {controlnet} does not exist in cache directory {self.engine_cache_dir}, it will be downloaded."
            )

        self.start_keepalive()
        result = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=torch.half,
            cache_dir=self.engine_cache_dir,
        )
        self.stop_keepalive()
        return result

    @property
    def controlnet(self) -> Optional[ControlNetModel]:
        """
        Gets the configured controlnet (or none.)
        """
        if not hasattr(self, "_controlnet"):
            self._controlnet = self.get_controlnet(self.controlnet_name)
        return self._controlnet

    @controlnet.setter
    def controlnet(
        self,
        new_controlnet: Optional[Literal["canny", "tile", "mlsd", "hed", "scribble", "inpaint"]],
    ) -> None:
        """
        Sets a new controlnet.
        """
        pretrained_path = None
        if new_controlnet == "canny":
            pretrained_path = CONTROLNET_CANNY
        elif new_controlnet == "mlsd":
            pretrained_path = CONTROLNET_MLSD
        elif new_controlnet == "hed":
            pretrained_path = CONTROLNET_HED
        elif new_controlnet == "tile":
            pretrained_path = CONTROLNET_TILE
        elif new_controlnet == "scribble":
            pretrained_path = CONTROLNET_SCRIBBLE
        elif new_controlnet == "inpaint":
            pretrained_path = CONTROLNET_INPAINT
        if pretrained_path is None and new_controlnet is not None:
            logger.error(f"Unsupported controlnet {new_controlnet}")

        existing_controlnet = getattr(self, "_controlnet", None)

        if (
            (existing_controlnet is None and new_controlnet is not None)
            or (existing_controlnet is not None and new_controlnet is None)
            or (existing_controlnet is not None and self.controlnet_name != new_controlnet)
        ):
            del self.pipeline
            if new_controlnet is not None:
                self._controlnet_name = new_controlnet
                self._controlnet = self.get_controlnet(pretrained_path)
            else:
                self._controlnet_name = None  # type: ignore
                self._controlnet = None

    @property
    def controlnet_name(self) -> Optional[str]:
        """
        Gets the name of the control net, if one was set.
        """
        if not hasattr(self, "_controlnet_name"):
            self._controlnet_name = self.configuration.get("enfugue.controlnet", None)
        return self._controlnet_name

    def check_download_checkpoint(self, remote_url: str) -> str:
        """
        Downloads a checkpoint directly to the checkpoints folder.
        """
        output_file = os.path.basename(remote_url)
        output_path = os.path.join(self.engine_checkpoints_dir, output_file)
        self.start_keepalive()
        check_download(remote_url, output_path)
        self.stop_keepalive()
        return output_path

    def __call__(self, **kwargs: Any) -> Any:
        """
        Passes an invocation down to the pipeline, doing whatever it needs to do to initialize it.
        Will switch between inpainting and non-inpainting models
        """
        logger.debug(f"Calling pipeline with arguments {kwargs}")

        if kwargs.get("mask", None) is not None:
            self.inpainting = True
        else:
            self.inpainting = False

        called_width = kwargs.get("width", self.size)
        called_height = kwargs.get("height", self.size)
        chunk_size = kwargs.get("chunking_size", self.chunking_size)
        if called_width < self.size:
            self.tensorrt_is_enabled = False
            logger.info(
                f"Width ({called_width}) less than configured width ({self.size}), disabling TensorRT"
            )
        elif called_height < self.size:
            logger.info(
                f"Height ({called_height}) less than configured height ({self.size}), disabling TensorRT"
            )
            self.tensorrt_is_enabled = False
        elif (called_width != self.size or called_height != self.size) and not chunk_size:
            logger.info(
                f"Dimensions do not match size of engine and chunking is disabled, disabling TensorRT"
            )
            self.tensorrt_is_enabled = False
        else:
            self.tenssort_is_enabled = True

        result = self.pipeline(generator=self.generator, **kwargs)
        if self.refiner is not None:
            if self.refiner_switch_mode == "offload":
                logger.debug(f"Sending pipeline to CPU")
                self.offload_pipeline()
                self.reload_refiner()
            elif self.refiner_switch_mode == "delete":
                logger.debug("Deleting pipeline to switch to refiner")
                self.unload_pipeline()
            else:
                logger.debug("Keeping pipeline in memory")

            for i, image in enumerate(result["images"]):
                is_nsfw = result["nsfw_content_detected"][i]
                if is_nsfw:
                    logger.info(f"Result {i} has NSFW content, not refining.")
                    continue
                logger.debug(f"Refining result {i}")
                kwargs.pop("image", None) # Remove any previous image
                kwargs.pop("mask", None) # Remove any previous mask
                kwargs.pop("strength", None) # Remove any previous strength
                result["images"][i] = self.refiner_pipeline(
                    generator=self.generator,
                    image=image,
                    strength=kwargs.pop("refiner_strength", self.refiner_strength),
                    guidance_scale=kwargs.pop("refiner_guidance_scale", self.refiner_guidance_scale),
                    aesthetic_score=kwargs.pop("aesthetic_score", self.aesthetic_score),
                    negative_aesthetic_score=kwargs.pop("negative_aesthetic_score", self.negative_aesthetic_score),
                    **kwargs
                )["images"][0]
            if self.refiner_switch_mode == "offload":
                logger.debug(f"Sending refiner to CPU and reloading pipeline")
                self.offload_refiner()
                self.reload_pipeline()
            elif self.refiner_switch_mode == "delete":
                logger.debug("Deleting refiner pipeline")
                self.unload_refiner()
            else:
                logger.debug("Keeping refiner in memory")

        self.tensorrt_is_enabled = True
        return result

    def write_tensorrt_metadata(self, path: str) -> None:
        """
        Writes metadata for TensorRT to a json file
        """
        if "controlnet" in path:
            dump_json(path, {"size": self.size, "controlnet": self.controlnet_name})
        else:
            dump_json(
                path,
                {
                    "size": self.size,
                    "lora": self.lora_names_weights,
                    "lycoris": self.lycoris_names_weights,
                    "inversion": self.inversion_names,
                },
            )

    @staticmethod
    def get_tensorrt_status(
        engine_root: str,
        model: str,
        size: Optional[int] = None,
        lora: Optional[Union[str, Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        lycoris: Optional[Union[str, Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        controlnet: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, bool]:
        """
        Gets the TensorRT status for an individual model.
        """
        tensorrt_is_supported = False

        try:
            import tensorrt

            tensorrt.__version__  # quiet importchecker
            tensorrt_is_supported = True
        except Exception as ex:
            logger.info("TensorRT is disabled.")
            logger.debug("{0}: {1}".format(type(ex).__name__, ex))
            pass

        if not tensorrt_is_supported:
            return {"supported": False, "ready": False}

        if model.endswith(".ckpt") or model.endswith(".safetensors"):
            model, _ = os.path.splitext(os.path.basename(model))
        else:
            model = os.path.basename(model)

        if model.endswith("-inpainting"):
            # Look for the base model instead, we'll look for inpainting separately
            model, _, _ = model.rpartition("-")

        model_dir = os.path.join(engine_root, "tensorrt", model)
        inpaint_model_dir = os.path.join(engine_root, "tensorrt", f"{model}-inpainting")

        if size is None:
            size = DiffusionPipelineManager.DEFAULT_SIZE

        if inversion is None:
            inversion = []
        elif not isinstance(inversion, list):
            inversion = [inversion]

        inversion_key = []
        for inversion_part in inversion:
            inversion_name, ext = os.path.splitext(os.path.basename(inversion_part))
            inversion_key.append(inversion_name)

        if lora is None:
            lora = []
        elif not isinstance(lora, list):
            lora = [lora]

        lora_key = []
        for lora_part in lora:
            if isinstance(lora_part, tuple):
                lora_path, lora_weight = lora_part
            else:
                lora_path, lora_weight = lora_part, 1.0
            lora_name, ext = os.path.splitext(os.path.basename(lora_path))
            lora_key.append((lora_name, lora_weight))

        if lora is None:
            lora = []
        elif not isinstance(lora, list):
            lora = [lora]

        lycoris_key = []
        for lycoris_part in lycoris:
            if isinstance(lycoris_part, tuple):
                lycoris_path, lycoris_weight = lycoris_part
            else:
                lycoris_path, lycoris_weight = lycoris_part, 1.0
            lycoris_name, ext = os.path.splitext(os.path.basename(lycoris_path))
            lycoris_key.append((lycoris_name, lycoris_weight))

        clip_ready = "clip" not in DiffusionPipelineManager.TENSORRT_STAGES
        vae_ready = "vae" not in DiffusionPipelineManager.TENSORRT_STAGES
        unet_ready = "unet" not in DiffusionPipelineManager.TENSORRT_STAGES
        inpaint_unet_ready = unet_ready
        controlled_unet_ready = unet_ready
        controlnet_ready: Union[bool, Dict[str, bool]] = (
            "controlnet" not in DiffusionPipelineManager.TENSORRT_STAGES
        )

        if not clip_ready:
            clip_key = DiffusionPipelineManager.get_tensorrt_clip_key(
                size, lora=lora_key, lycoris=lycoris_key, inversion=inversion_key
            )
            clip_plan = os.path.join(model_dir, "clip", clip_key, "engine.plan")
            clip_ready = os.path.exists(clip_plan)

        if not vae_ready:
            vae_key = DiffusionPipelineManager.get_tensorrt_vae_key(
                size, lora=lora_key, lycoris=lycoris_key, inversion=inversion_key
            )
            vae_plan = os.path.join(model_dir, "vae", vae_key, "engine.plan")
            vae_ready = os.path.exists(vae_plan)

        if not unet_ready:
            unet_key = DiffusionPipelineManager.get_tensorrt_unet_key(
                size, lora=lora_key, lycoris=lycoris_key, inversion=inversion_key
            )
            unet_plan = os.path.join(model_dir, "unet", unet_key, "engine.plan")
            unet_ready = os.path.exists(unet_plan)

            inpaint_unet_plan = os.path.join(inpaint_model_dir, "unet", unet_key, "engine.plan")
            inpaint_unet_ready = os.path.exists(inpaint_unet_plan)

            controlled_unet_key = DiffusionPipelineManager.get_tensorrt_controlled_unet_key(
                size, lora=lora_key, lycoris=lycoris_key, inversion=inversion_key
            )
            controlled_unet_plan = os.path.join(
                model_dir, "controlledunet", controlled_unet_key, "engine.plan"
            )
            controlled_unet_ready = os.path.exists(controlled_unet_plan)

        if not controlnet_ready:
            if controlnet is None:
                controlnet_ready = True
            else:
                controlnet_ready = {}
                if not isinstance(controlnet, list):
                    controlnet = [controlnet]
                for controlnet_name in controlnet:
                    controlnet_key = DiffusionPipelineManager.get_tensorrt_controlnet_key(
                        size, controlnet=controlnet_name
                    )
                    controlnet_plan = os.path.join(
                        model_dir, "controlnet", controlnet_key, "engine.plan"
                    )
                    controlnet_ready[controlnet_name] = os.path.exists(controlnet_plan)

        ready = clip_ready and vae_ready
        if controlnet is not None or DiffusionPipelineManager.TENSORRT_ALWAYS_USE_CONTROLLED_UNET:
            ready = ready and controlled_unet_ready
            if isinstance(controlnet_ready, dict):
                for name in controlnet_ready:
                    ready = ready and controlnet_ready[name]
        else:
            ready = ready and unet_ready

        return {
            "supported": tensorrt_is_supported,
            "unet_ready": unet_ready,
            "controlled_unet_ready": controlled_unet_ready,
            "vae_ready": vae_ready,
            "clip_ready": clip_ready,
            "inpaint_unet_ready": inpaint_unet_ready,
            "ready": ready,
        }

    def create_inpainting_checkpoint(
        self, source_checkpoint_path: str, target_checkpoint_path: str
    ) -> None:
        """
        Creates an inpainting model by merging in the SD 1.5 inpainting model with a non inpainting model.
        """
        from enfugue.diffusion.util import ModelMerger

        self.start_keepalive()
        try:
            merger = ModelMerger(
                self.check_download_checkpoint(DEFAULT_INPAINTING_MODEL),
                source_checkpoint_path,
                self.check_download_checkpoint(DEFAULT_MODEL),
                interpolation="add-difference",
            )
            merger.save(target_checkpoint_path)
        except Exception as ex:
            logger.error(
                f"Couldn't save merged checkpoint made from {source_checkpoint_path} to {target_checkpoint_path}: {ex}"
            )
            logger.error(traceback.format_exc())
            raise
        else:
            logger.info(f"Saved merged inpainting checkpoint at {target_checkpoint_path}")
        self.stop_keepalive()
