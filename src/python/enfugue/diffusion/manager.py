from __future__ import annotations

import os
import time
import torch
import random
import datetime
import threading

from typing import (
    Type,
    Union,
    Any,
    Optional,
    List,
    Tuple,
    Dict,
    Literal,
    Callable,
    TYPE_CHECKING,
    cast,
)
from hashlib import md5

from pibble.api.configuration import APIConfiguration
from pibble.api.exceptions import ConfigurationError
from pibble.util.files import load_json, dump_json

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
    def device(self) -> str:
        """
        Gets the device that will be executed on
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

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
            self._generator = torch.Generator(device=self.device)
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
        root = self.configuration.get("enfugue.engine.root", "~/.cache/enfugue")
        if root.startswith("~"):
            root = os.path.expanduser(root)
        root = os.path.realpath(root)
        check_make_directory(root)
        return root

    @property
    def diffusers_cache_dir(self) -> str:
        """
        Gets the cache for diffusers-downloaded configuration files, base models, etc.
        """
        path = os.path.join(self.engine_root, "cache")
        check_make_directory(path)
        return path

    @property
    def engine_checkpoints_dir(self) -> str:
        """
        Gets where checkpoints are downloaded in.
        """
        path = os.path.join(self.engine_root, "checkpoint")
        check_make_directory(path)
        return path

    @property
    def engine_other_dir(self) -> str:
        """
        Gets where any other weights are download in
        """
        path = os.path.join(self.engine_root, "other")
        check_make_directory(path)
        return path

    @property
    def engine_lora_dir(self) -> str:
        """
        Gets where lora are downloaded in.
        """
        path = os.path.join(self.engine_root, "lora")
        check_make_directory(path)
        return path

    @property
    def engine_inversions_dir(self) -> str:
        """
        Gets where inversions are downloaded in.
        """
        path = os.path.join(self.engine_root, "inversions")
        check_make_directory(path)
        return path

    @property
    def engine_models_dir(self) -> str:
        """
        Gets where models are pushed to after converting to diffusers.
        """
        path = os.path.join(self.engine_root, "models")
        check_make_directory(path)
        return path

    @property
    def model_dir(self) -> str:
        """
        Gets where the current model should be stored.
        """
        model = self.model
        path = os.path.join(self.engine_models_dir, self.model)
        check_make_directory(path)
        return path

    @property
    def model_tensorrt_dir(self) -> str:
        """
        Gets where tensorrt engines will be built per model.
        """
        path = os.path.join(self.model_dir, "tensorrt")
        check_make_directory(path)
        return path

    @staticmethod
    def get_tensorrt_clip_key(
        size: int, lora: List[Tuple[str, float]], inversion: List[str], **kwargs: Any
    ) -> str:
        """
        Uses hashlib to generate the unique key for the CLIP engine.
        CLIP must be rebuilt for each:
            1. Model
            2. Dimension
            3. LoRA
            4. Textual Inversion
        """
        return md5(
            "-".join(
                [
                    str(size),
                    ":".join(
                        "=".join([str(part) for part in lora_weight])
                        for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
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

    @staticmethod
    def get_tensorrt_unet_key(
        size: int,
        lora: List[Tuple[str, float]],
        inversion: List[str],
        **kwargs: Any,
    ) -> str:
        """
        Uses hashlib to generate the unique key for the UNET engine.
        UNET must be rebuilt for each:
            1. Model
            2. Dimension
            3. LoRA
            4. Textual Inversion
        """
        return md5(
            "-".join(
                [
                    str(size),
                    ":".join(
                        "=".join([str(part) for part in lora_weight])
                        for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
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
            inversion=self.inversion_names,
        )

    @property
    def model_tensorrt_unet_dir(self) -> str:
        """
        Gets where the tensorrt CLIP engine will be stored.
        """
        path = os.path.join(self.model_tensorrt_dir, "unet", self.model_tensorrt_unet_key)
        check_make_directory(path)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            self.write_tensorrt_metadata(metadata_path)
        return path

    @staticmethod
    def get_tensorrt_controlled_unet_key(
        size: int,
        lora: List[Tuple[str, float]],
        inversion: List[str],
        **kwargs: Any,
    ) -> str:
        """
        Uses hashlib to generate the unique key for the UNET engine with controlnet blocks.
        ControlledUNET must be rebuilt for each:
            1. Model
            2. Dimension
            3. LoRA
            4. Textual Inversion
        """
        return md5(
            "-".join(
                [
                    str(size),
                    ":".join(
                        "=".join([str(part) for part in lora_weight])
                        for lora_weight in sorted(lora, key=lambda lora_part: lora_part[0])
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
        if new_enabled != self.tensorrt_is_enabled:
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
            if self._model.endswith(".ckpt") or self._model.endswith(".safetensors"):
                self._model = self.check_convert_checkpoint(self._model)
        return self._model

    @model.setter
    def model(self, new_model: Optional[str]) -> None:
        """
        Sets a new model. Destroys the pipeline.
        """
        if new_model is None:
            new_model = self.configuration.get("enfugue.model", DEFAULT_MODEL)
        if new_model.endswith(".ckpt") or new_model.endswith(".safetensors"):
            new_model = self.check_convert_checkpoint(new_model)
        if self.model != new_model:
            del self.pipeline
        self._model = new_model

    @property
    def dtype(self) -> torch.dtype:
        if not hasattr(self, "_torch_dtype"):
            if self.device == "cpu":
                logger.debug("Inferencing on CPU, using BFloat")
                self._torch_dtype = torch.float
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
        if self.device == "cpu":
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
    def inversion(self) -> List[str]:
        """
        Get textual inversions added to the text encoder.
        """
        return getattr(self, "_inversion", [])

    @inversion.setter
    def inversion(self, new_inversion: Optional[Union[str, List[str]]]) -> None:
        """
        Sets new textual inversions. Destroys the pipeline.
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
    def model_ckpt_path(self) -> str:
        """
        Gets the path for a model checkpoint. May be a .ckpt or .safetensors.
        """
        ckpt_path = os.path.join(self.engine_checkpoints_dir, f"{self.model}.ckpt")
        safetensor_path = os.path.join(self.engine_checkpoints_dir, f"{self.model}.safetensors")

        if os.path.exists(ckpt_path):
            return ckpt_path
        elif os.path.exists(safetensor_path):
            return safetensor_path
        else:
            raise IOError(f"Unknown model {self.model}")

    @property
    def vae_config(self) -> Dict:
        """
        Reads the VAE config.json file.
        """
        path = os.path.join(self.model_dir, "vae", "config.json")
        if not os.path.exists(path):
            raise OSError(f"Couldn't find VAE config file at {path}")
        return cast(Dict, load_json(path))

    @property
    def unet_config(self) -> Dict:
        """
        Reads the unet config.json file.
        """
        path = os.path.join(self.model_dir, "unet", "config.json")
        if not os.path.exists(path):
            raise OSError(f"Couldn't find unet config file at {path}")
        return cast(Dict, load_json(path))

    @property
    def clip_config(self) -> Dict:
        """
        Reads the clip config.json file.
        """
        path = os.path.join(self.model_dir, "text_encoder", "config.json")
        if not os.path.exists(path):
            raise OSError(f"Couldn't find clip config file at {path}")
        return cast(Dict, load_json(path))

    @property
    def inpainting(self) -> bool:
        """
        Returns true if the model is an inpainting model.
        """
        return self.unet_config["in_channels"] == 9

    @inpainting.setter
    def inpainting(self, new_inpainting: bool) -> None:
        """
        Sets whether or not we are inpainting.

        We trade efficiency for ease-of-use here; we just keep a model named `-inpainting`
        for any model.
        """
        if self.inpainting != new_inpainting:
            del self.pipeline
            current_checkpoint_path = self.model_ckpt_path

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
    def pipeline(self) -> EnfugueStableDiffusionPipeline:
        """
        Instantiates the pipeline.
        """
        if not hasattr(self, "_pipeline"):
            kwargs = {
                "cache_dir": self.diffusers_cache_dir,
                "engine_size": self.size,
                "chunking_size": self.chunking_size,
                "requires_safety_checker": self.safe,
                "torch_dtype": self.dtype,
            }

            if not self.safe:
                kwargs["safety_checker"] = None

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

            logger.debug(
                f"Initializing pipeline model {self.model} in directory {self.model_dir} with arguments {kwargs}"
            )
            kwargs["controlnet"] = self.controlnet

            pipeline = self.pipeline_class.from_pretrained(self.model_dir, **kwargs)
            if not self.tensorrt_is_ready:
                for lora, weight in self.lora:
                    logger.debug(f"Adding LoRA {lora} to pipeline")
                    pipeline.load_lora_weights(lora, multiplier=weight)
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
            logger.debug("Pipeline delete called, but no pipeline present.")

    def unload_pipeline(self) -> None:
        """
        Calls the pipeline deleter.
        """
        del self.pipeline

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

        self.start_keepalive()
        result = ControlNetModel.from_pretrained(
            controlnet,
            torch_dtype=torch.half,
            cache_dir=self.diffusers_cache_dir,
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

    def check_convert_checkpoint(self, checkpoint_path: str) -> str:
        """
        Converts a .ckpt file to the directory structure from diffusers
        """
        checkpoint_file = os.path.basename(checkpoint_path)
        model_name, ext = os.path.splitext(checkpoint_file)
        model_dir = os.path.join(self.engine_models_dir, model_name)

        if not os.path.exists(model_dir):
            if checkpoint_path.startswith("http"):
                checkpoint_path = self.check_download_checkpoint(checkpoint_path)

            from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
                download_from_original_stable_diffusion_ckpt,
            )

            self.start_keepalive()
            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path=checkpoint_path,
                scheduler_type="ddim",
                from_safetensors=ext == ".safetensors",
                num_in_channels=9 if "inpaint" in checkpoint_path else 4,
            ).to(torch_dtype=self.dtype)
            pipe.save_pretrained(model_dir)
            self.stop_keepalive()
        return model_name

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
                    "inversion": self.inversion_names,
                },
            )

    @staticmethod
    def get_tensorrt_status(
        engine_root: str,
        model: str,
        size: Optional[int] = None,
        lora: Optional[Union[str, Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
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

        model_dir = os.path.join(engine_root, "models", model)
        inpaint_model_dir = os.path.join(engine_root, "models", f"{model}-inpainting")

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
                size, lora=lora_key, inversion=inversion_key
            )
            clip_plan = os.path.join(model_dir, "tensorrt", "clip", clip_key, "engine.plan")
            clip_ready = os.path.exists(clip_plan)

        if not vae_ready:
            vae_key = DiffusionPipelineManager.get_tensorrt_vae_key(
                size, lora=lora_key, inversion=inversion_key
            )
            vae_plan = os.path.join(model_dir, "tensorrt", "vae", vae_key, "engine.plan")
            vae_ready = os.path.exists(vae_plan)

        if not unet_ready:
            unet_key = DiffusionPipelineManager.get_tensorrt_unet_key(
                size, lora=lora_key, inversion=inversion_key
            )
            unet_plan = os.path.join(model_dir, "tensorrt", "unet", unet_key, "engine.plan")
            unet_ready = os.path.exists(unet_plan)

            inpaint_unet_plan = os.path.join(
                inpaint_model_dir, "tensorrt", "unet", unet_key, "engine.plan"
            )
            inpaint_unet_ready = os.path.exists(inpaint_unet_plan)

            controlled_unet_key = DiffusionPipelineManager.get_tensorrt_controlled_unet_key(
                size, lora=lora_key, inversion=inversion_key
            )
            controlled_unet_plan = os.path.join(
                model_dir, "tensorrt", "controlledunet", controlled_unet_key, "engine.plan"
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
                        model_dir, "tensorrt", "controlnet", controlnet_key, "engine.plan"
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
        merger = ModelMerger(
            self.check_download_checkpoint(DEFAULT_INPAINTING_MODEL),
            source_checkpoint_path,
            self.check_download_checkpoint(DEFAULT_MODEL),
            interpolation="add-difference",
        )
        merger.save(target_checkpoint_path)
        logger.info(f"Saved merged inpainting checkpoint at {target_checkpoint_path}")
        self.stop_keepalive()
