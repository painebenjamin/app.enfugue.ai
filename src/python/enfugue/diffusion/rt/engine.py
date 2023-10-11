from __future__ import annotations

import os
import gc
import onnx
import torch
import logging
import tensorrt as trt

from copy import copy
from typing import Dict, Optional, Tuple, Any, List, Union

from polygraphy import cuda
from collections import OrderedDict

from polygraphy.logger import G_LOGGER
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig as TRTCreateConfig
from polygraphy.backend.trt import Profile as TRTProfile
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)

from enfugue.util import logger
from enfugue.diffusion.util import DTypeConverter
from enfugue.diffusion.rt.model import BaseModel


class Engine:
    """
    This class manages state for TensorRT engines.

    Building a TRT engine is a three-step process:
    1. Generate ONNX model from NN module
    2. Optimize ONNX model
    3. Compile TRT engine from optimized ONNX model
    """

    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    buffers: OrderedDict[str, cuda.DeviceView]
    tensors: OrderedDict[str, torch.Tensor]

    def __init__(self, engine_path: str) -> None:
        """ """
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.hijack_logger()

    def hijack_logger(self) -> None:
        """
        Sets the TRT logger to use the enfugue logger.
        """

        def _g_logger_log(msg: str, severity: int = logging.DEBUG, **kwargs: Any) -> None:
            if isinstance(msg, str) and msg.startswith("Total # of Profiles"):
                return  # Ignore this log spam
            if severity not in [
                logging.DEBUG,
                logging.INFO,
                logging.ERROR,
                logging.CRITICAL,
                logging.WARNING,
            ]:
                severity = logging.DEBUG
            logger.log(severity, msg)

        def _trt_logger_log(severity: trt.tensorrt.ILogger.Severity, msg: str) -> None:
            converted_severity = {
                trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
                trt.Logger.ERROR: logging.ERROR,
                trt.Logger.WARNING: logging.WARNING,
                trt.Logger.INFO: logging.INFO,
                trt.Logger.VERBOSE: logging.DEBUG,
            }.get(severity, logging.DEBUG)
            logger.log(converted_severity, msg)

        G_LOGGER.log = _g_logger_log
        trt_util.get_trt_logger().log = _trt_logger_log

    def __del__(self) -> None:
        """
        Deletes the engine.
        """
        for buf in self.buffers.values():
            try:
                buf.free()
            except:
                pass
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def load(self) -> None:
        """
        Loads the engine.
        """
        logger.debug(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self) -> None:
        """
        Creates execution context from the engine.
        """
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict: Optional[Dict[str, Tuple]] = None, device: Union[str, torch.device] = "cuda"):
        """
        Allocates TRT buffers.
        """
        bindings_per_profile = trt_util.get_bindings_per_profile(self.engine)

        for i in range(bindings_per_profile):
            binding = self.engine[i]

            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(i, shape)

            tensor = torch.empty(tuple(shape), dtype=DTypeConverter.from_numpy(dtype)).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)
            logger.debug(f"Binding {binding} to tensor of shape {shape}")

    def infer(self, feeds: Dict[str, cuda.DeviceView], stream: cuda.Stream) -> OrderedDict[str, torch.Tensor]:
        """
        Runs inference through the engine.
        """
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)

        # Shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feeds.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf

        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not success:
            raise IOError("Inference failed!")

        return self.tensors

    @staticmethod
    def get_onnx_path(engine_root: str, opt: bool = False) -> str:
        """
        Gets the path for the ONNX model in a directory.
        """
        model_name = f"model.opt" if opt else "model"
        return os.path.join(engine_root, f"{model_name}.onnx")

    @staticmethod
    def get_timing_cache(engine_root: str) -> str:
        """
        Gets the path for the timing cache in a directory.
        """
        return os.path.join(engine_root, "timing_cache")

    @staticmethod
    def get_engine_path(engine_root: str) -> str:
        """
        Gets the path for the engine in a directory.
        """
        return os.path.join(engine_root, "engine.plan")

    @staticmethod
    def get_paths(engine_root: str) -> Tuple[str, str, str, str]:
        """
        Gets the four paths; timing cache, engine path, model, and opt model.
        """
        return (
            Engine.get_timing_cache(engine_root),
            Engine.get_engine_path(engine_root),
            Engine.get_onnx_path(engine_root),
            Engine.get_onnx_path(engine_root, True),
        )

    def build(
        self,
        onnx_path: str,
        use_fp16: bool,
        input_profile: Optional[Dict[str, List[Tuple[int, ...]]]] = None,
        enable_preview: bool = False,
        enable_all_tactics: bool = False,
        timing_cache: str = "timing_cache",
        workspace_size: int = 0,
    ) -> None:
        """
        Builds a TRT engine.
        """
        dtype = "fp16" if use_fp16 else "fp32"
        logger.debug(f"Building TensorRT engine for {onnx_path}: {self.engine_path} ({dtype})")
        profile = TRTProfile()
        if input_profile:
            for name, dims in input_profile.items():
                profile.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs: dict[str, Any] = {}

        config_kwargs["preview_features"] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs["preview_features"].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        config = TRTCreateConfig(
            fp16=use_fp16,
            profiles=[profile],
            load_timing_cache=timing_cache,
            **config_kwargs,
        )
        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=config,
            save_timing_cache=timing_cache,
        )
        save_engine(engine, self.engine_path)

    @staticmethod
    def build_all(
        models: Dict[str, BaseModel],
        onnx_opset: int,
        opt_image_height: int,
        opt_image_width: int,
        opt_batch_size: int = 1,
        use_fp16: bool = False,
        force_engine_rebuild: bool = False,
        static_batch: bool = False,
        static_shape: bool = True,
        enable_preview: bool = False,
        enable_all_tactics: bool = False,
        max_workspace_size: int = 0,
    ) -> Dict[str, Engine]:
        """
        Builds all TRT engines at once.
        """
        built_engines = {}

        # Export models to ONNX
        for model_dir, model_obj in models.items():
            model_name = model_obj.get_model_key()
            timing_cache, engine_path, onnx_path, onnx_opt_path = Engine.get_paths(model_dir)
            if force_engine_rebuild or not os.path.exists(engine_path):
                logger.debug(f"Building engine for {model_name} at {model_dir}")
                if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                    if force_engine_rebuild or not os.path.exists(onnx_path):
                        logger.debug(f"Exporting model to {onnx_path}")
                        model = model_obj.get_model()
                        model.to("cuda")
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=onnx_opset,
                                do_constant_folding=True,
                                input_names=model_obj.get_input_names(),
                                output_names=model_obj.get_output_names(),
                                dynamic_axes=model_obj.get_dynamic_axes(),
                            )
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        logger.debug(f"Found cached model at {onnx_path}")

                    # Optimize onnx
                    if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                        logger.debug(f"Generating optimized model to {onnx_opt_path}")
                        onnx_opt_graph = model_obj.optimize(onnx.load(onnx_path))
                        onnx.save_model(onnx_opt_graph, onnx_opt_path)
                    else:
                        logger.debug(f"Found cached optimized model at {onnx_opt_path}")

        # Build TensorRT engines
        for model_dir, model_obj in models.items():
            model_name = model_obj.get_model_key()
            timing_cache, engine_path, onnx_path, onnx_opt_path = Engine.get_paths(model_dir)
            engine = Engine(engine_path)
            logger.debug(
                model_obj.get_input_profile(
                    opt_batch_size,
                    opt_image_height,
                    opt_image_width,
                    static_batch=static_batch,
                    static_shape=static_shape,
                )
            )

            if force_engine_rebuild or not os.path.exists(engine.engine_path):
                logger.debug(f"Building TensorRT engine for {model_name}")
                engine.build(
                    onnx_opt_path,
                    use_fp16=use_fp16,
                    input_profile=model_obj.get_input_profile(
                        opt_batch_size,
                        opt_image_height,
                        opt_image_width,
                        static_batch=static_batch,
                        static_shape=static_shape,
                    ),
                    enable_preview=enable_preview,
                    timing_cache=timing_cache,
                    workspace_size=max_workspace_size,
                )
            built_engines[model_name] = engine

        # Load and activate TensorRT engines
        for model_dir, model_obj in models.items():
            model_name = model_obj.get_model_key()
            engine = built_engines[model_name]
            engine.load()
            engine.activate()

        return built_engines
