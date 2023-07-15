from __future__ import annotations

import os
import gc
import onnx
import torch
import logging

from copy import copy
from typing import Dict, Union, Type, Any

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from enfugue.diffusion.pipeline import EnfugueStableDiffusionPipeline
from enfugue.diffusion.util import DTypeConverter
from enfugue.diffusion.ox.model import (
    BaseModel,
    CLIP,
    UNet,
    ControlledUNet,
    ControlNet,
    VAE
)
from enfugue.util import logger

class ONNXBuilder:
    """
    This class manages state for ONNX engines
    """

    models: Dict[str, BaseModel]

    def __init__(
        self,
        embedding_dim: int,
        max_batch_size: int,
        device: torch.device,
        onnx_opset: int,
        use_fp16: bool = False
    ) -> None:
        self.embedding_dim = embedding_dim
        self.max_batch_size = max_batch_size
        self.device = device
        self.onnx_opset = onnx_opset
        self.use_fp16 = use_fp16
        self.models = {}

    def add_model(
        self,
        directory: str,
        model_type: Type,
        module: torch.nn.Module,
        **kwargs: Any
    ) -> None:
        self.models[directory] = model_type(
            module,
            embedding_dim=self.embedding_dim,
            max_batch_size=self.max_batch_size,
            use_fp16=self.use_fp16,
            device=self.device,
            **kwargs
        )

    def add_text_encoder(
        self,
        directory: str, 
        text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
    ) -> None:
        self.add_model(
            directory=directory,
            model_type=CLIP,
            module=text_encoder
        )
    
    def add_unet(
        self,
        directory: str, 
        unet: UNet2DConditionModel,
        in_channels: int,
        is_controlled: bool = False
    ) -> None:
        self.add_model(
            directory=directory,
            model_type=ControlledUNet if is_controlled else UNet,
            module=unet,
            unet_dim=in_channels
        )
    
    def add_controlnet(
        self,
        directory: str, 
        controlnet: ControlNetModel,
        in_channels: int,
    ) -> None:
        self.add_model(
            directory=directory,
            model_type=ControlNet,
            module=controlnet,
            unet_dim=in_channels
        )

    def add_vae(
        self,
        directory: str, 
        vae: AutoencoderKL,
    ) -> None:
        self.add_model(
            directory=directory,
            model_type=VAE,
            module=vae,
        )
    
    def build_all(
        self,
        opt_image_height: int,
        opt_image_width: int,
        opt_batch_size: int = 1,
        rebuild: bool = False,
    ) -> None:
        """
        Builds all engines.
        """
        for model_dir, model_obj in self.models.items():
            model_name = model_obj.get_model_key()
            onnx_path = os.path.join(model_dir, "model.onnx")
            onnx_opt_path = os.path.join(model_dir, "model.opt.onnx")
            logger.debug(f"Checking for {model_name} in {model_dir} (using class {model_obj})")
            if rebuild or not os.path.exists(onnx_opt_path):
                if rebuild or not os.path.exists(onnx_path):
                    logger.debug(f"Exporting model to {onnx_path}")
                    model = model_obj.get_model()
                    with torch.autocast("cpu"), torch.no_grad():
                        inputs = model_obj.get_sample_input(
                            opt_batch_size, opt_image_height, opt_image_width
                        )
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=self.onnx_opset,
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
                if rebuild or not os.path.exists(onnx_opt_path):
                    logger.debug(f"Generating optimized model to {onnx_opt_path}")
                    onnx_opt_graph = model_obj.optimize(onnx_path)
                    onnx.save_model(onnx_opt_graph, onnx_opt_path)
                else:
                    logger.debug(f"Found cached optimized model at {onnx_opt_path}")
