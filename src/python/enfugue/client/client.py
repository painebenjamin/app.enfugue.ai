from __future__ import annotations

import os
import time

from typing import Dict, Any, Optional, Literal, TypedDict, List, Union
from PIL.Image import Image

from urllib.parse import urlparse

from pibble.api.client.webservice.jsonapi import JSONWebServiceAPIClient
from pibble.ext.user.client.base import UserExtensionClientBase

from enfugue.diffusion.plan import NodeDict
from enfugue.diffusion.constants import *
from enfugue.util import (
    logger,
    IMAGE_FIT_LITERAL,
    IMAGE_ANCHOR_LITERAL
)
from enfugue.client.invocation import RemoteInvocation

__all__ = ["WeightedModelDict", "EnfugueClient"]


class WeightedModelDict(TypedDict):
    """
    Represents a model and weight for LoRA and LyCORIS
    """

    model: str
    weight: float


class EnfugueClient(UserExtensionClientBase, JSONWebServiceAPIClient):
    """
    Extend the client base to add helper method calls.
    """
    def __init__(self) -> None:
        super(EnfugueClient, self).__init__()
        self.configuration.environment_prefix = "ENFUGUE"

    def configure(self, **configuration) -> None:
        """
        Intercept configure to add defaults
        """
        client = configuration.get("client", {})
        if "host" not in client:
            client["host"] = "app.enfugue.ai"
        if "port" not in client:
            client["port"] = 45554
        if "secure" not in client:
            client["secure"] = True
        configuration["client"] = client
        super(EnfugueClient, self).configure(**configuration)

    def on_configure(self) -> None:
        """
        On configuration, login.
        """
        username = self.configuration.get("enfugue.username", None)
        password = self.configuration.get("enfugue.password", None)
        if username and password:
            self.login(username, password)

    def checkpoints(self) -> List[str]:
        """
        Gets a list of installed checkpoints.
        """
        return self.get("/api/checkpoints").json()["data"]

    def lora(self) -> List[str]:
        """
        Gets a list of installed lora.
        """
        return self.get("/api/lora").json()["data"]

    def lycoris(self) -> List[str]:
        """
        Gets a list of installed lycoris.
        """
        return self.get("/api/lycoris").json()["data"]

    def inversion(self) -> List[str]:
        """
        Gets a list of installed inversion.
        """
        return self.get("/api/inversion").json()["data"]

    def download(
        self,
        download_type: Literal["checkpoint", "inversion", "lora", "lycoris"],
        url: str,
        filename: Optional[str] = None,
        polling_interval: int = 5,
        overwrite: bool = False,
    ) -> None:
        """
        Downloads a file, waiting for it to complete.
        """
        if filename is None:
            filename = os.path.basename(urlparse(url).path)
        data = {"type": download_type, "url": url, "filename": filename, "overwrite": overwrite}
        status = self.post("/api/download", data=data).json()["data"]
        while status["status"] != "complete":
            time.sleep(polling_interval)
            statuses = self.get("/api/download").json()["data"]
            for download_status in statuses:
                if download_status["filename"] == filename:
                    status = download_status
                    break

    def status(self) -> Dict[str, Any]:
        """
        Gets the status from the API.
        """
        return self.get("/api").json()["data"]

    def settings(self) -> Dict[str, Any]:
        """
        Gets settings from the remote server.
        """
        return self.get("/api/settings").json()["data"]

    def invoke(
        self,
        prompt: Optional[str] = None,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        model_prompt: Optional[str] = None,
        model_prompt_2: Optional[str] = None,
        model_negative_prompt: Optional[str] = None,
        model_negative_prompt_2: Optional[str] = None,
        intermediates: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        chunking_size: Optional[int] = None,
        chunking_blur: Optional[int] = None,
        samples: Optional[int] = None,
        iterations: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        refiner_strength: Optional[float] = None,
        refiner_guidance_scale: Optional[float] = None,
        refiner_aesthetic_score: Optional[float] = None,
        refiner_negative_aesthetic_score: Optional[float] = None,
        refiner_prompt: Optional[str] = None,
        refiner_prompt_2: Optional[str] = None,
        refiner_negative_prompt: Optional[str] = None,
        refiner_negative_prompt_2: Optional[str] = None,
        nodes: Optional[List[NodeDict]] = None,
        model: Optional[str] = None,
        model_type: Optional[Literal["checkpoint", "model"]] = None,
        size: Optional[int] = None,
        refiner_size: Optional[int] = None,
        inpainter_size: Optional[int] = None,
        inpainter: Optional[str] = None,
        refiner: Optional[str] = None,
        lora: Optional[List[WeightedModelDict]] = None,
        lycoris: Optional[List[WeightedModelDict]] = None,
        inversion: Optional[List[str]] = None,
        scheduler: Optional[SCHEDULER_LITERAL] = None,
        vae: Optional[str] = None,
        refiner_vae: Optional[str] = None,
        inpainter_vae: Optional[str] = None,
        seed: Optional[int] = None,
        image: Optional[Union[str, Image]] = None,
        mask: Optional[Union[str, Image]] = None,
        control_image: Optional[Union[str, Image]] = None,
        controlnet: Optional[CONTROLNET_LITERAL] = None,
        strength: Optional[float] = None,
        fit: Optional[IMAGE_FIT_LITERAL] = None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        remove_background: Optional[bool] = None,
        fill_background: Optional[bool] = None,
        scale_to_model_size: Optional[bool] = None,
        invert_control_image: Optional[bool] = None,
        invert_mask: Optional[bool] = None,
        process_control_image: Optional[bool] = True,
        conditioning_scale: Optional[float] = None,
        crop_inpaint: Optional[bool] = None,
        inpaint_feather: Optional[int] = None,
        outscale: Optional[int] = 1,
        upscale: Optional[Union[UPSCALE_LITERAL, List[UPSCALE_LITERAL]]] = None,
        upscale_pipeline: Optional[UPSCALE_PIPELINE_LITERAL] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = None,
        upscale_diffusion_guidance_scale: Optional[Union[Union[int, float], List[Union[int, float]]]] = None,
        upscale_diffusion_strength: Optional[Union[float, List[float]]] = None,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_negative_prompt: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_negative_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_controlnet: Optional[Union[CONTROLNET_LITERAL, List[CONTROLNET_LITERAL]]] = None,
        upscale_diffusion_chunking_size: Optional[int] = None,
        upscale_diffusion_chunking_blur: Optional[int] = None,
        upscale_diffusion_scale_chunking_size: Optional[bool] = None,
        upscale_diffusion_scale_chunking_blur: Optional[bool] = None,
    ) -> RemoteInvocation:
        """
        Invokes the engine.
        """
        kwargs: Dict[str, Any] = {}

        if model is not None:
            kwargs["model"] = model
        if model_type is not None:
            kwargs["model_type"] = model_type
        if prompt is not None:
            kwargs["prompt"] = prompt
        if prompt_2 is not None:
            kwargs["prompt_2"] = prompt_2
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if negative_prompt_2 is not None:
            kwargs["negative_prompt_2"] = negative_prompt_2
        if model_prompt is not None:
            kwargs["model_prompt"] = model_prompt
        if model_prompt_2 is not None:
            kwargs["model_prompt_2"] = model_prompt_2
        if model_negative_prompt is not None:
            kwargs["model_negative_prompt"] = model_negative_prompt
        if model_negative_prompt_2 is not None:
            kwargs["model_negative_prompt_2"] = model_negative_prompt_2
        if intermediates is not None:
            kwargs["intermediates"] = intermediates
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if chunking_size is not None:
            kwargs["chunking_size"] = chunking_size
        if chunking_blur is not None:
            kwargs["chunking_blur"] = chunking_blur
        if samples is not None:
            kwargs["samples"] = samples
        if iterations is not None:
            kwargs["iterations"] = iterations
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if refiner_strength is not None:
            kwargs["refiner_strength"] = refiner_strength
        if refiner_guidance_scale is not None:
            kwargs["refiner_guidance_scale"] = refiner_guidance_scale
        if refiner_aesthetic_score is not None:
            kwargs["refiner_aesthetic_score"] = refiner_aesthetic_score
        if refiner_negative_aesthetic_score is not None:
            kwargs["refiner_negative_aesthetic_score"] = refiner_negative_aesthetic_score
        if refiner_prompt is not None:
            kwargs["refiner_prompt"] = refiner_prompt
        if refiner_negative_prompt is not None:
            kwargs["refiner_negative_prompt"] = refiner_negative_prompt
        if nodes is not None:
            kwargs["nodes"] = nodes
        if size is not None:
            kwargs["size"] = size
        if refiner_size is not None:
            kwargs["refiner_size"] = refiner_size
        if inpainter_size is not None:
            kwargs["inpainter_size"] = inpainter_size
        if inpainter is not None:
            kwargs["inpainter"] = inpainter
        if refiner is not None:
            kwargs["refiner"] = refiner
        if lora is not None:
            kwargs["lora"] = lora
        if lycoris is not None:
            kwargs["lycoris"] = lycoris
        if scheduler is not None:
            kwargs["scheduler"] = scheduler
        if vae is not None:
            kwargs["vae"] = vae
        if refiner_vae is not None:
            kwargs["refiner_vae"] = refiner_vae
        if refiner_vae is not None:
            kwargs["refiner_vae"] = refiner_vae
        if seed is not None:
            kwargs["seed"] = seed
        if image is not None:
            kwargs["image"] = image
        if mask is not None:
            kwargs["mask"] = mask
        if control_image is not None:
            kwargs["control_image"] = control_image
        if controlnet is not None:
            kwargs["controlnet"] = controlnet
        if strength is not None:
            kwargs["strength"] = strength
        if fit is not None:
            kwargs["fit"] = fit
        if anchor is not None:
            kwargs["anchor"] = anchor
        if remove_background is not None:
            kwargs["remove_background"] = remove_background
        if fill_background is not None:
            kwargs["fill_background"] = fill_background
        if scale_to_model_size is not None:
            kwargs["scale_to_model_size"] = scale_to_model_size
        if invert_control_image is not None:
            kwargs["invert_control_image"] = invert_control_image
        if invert_mask is not None:
            kwargs["invert_mask"] = invert_mask
        if process_control_image is not None:
            kwargs["process_control_image"] = process_control_image
        if conditioning_scale is not None:
            kwargs["conditioning_scale"] = conditioning_scale
        if crop_inpaint is not None:
            kwargs["crop_inpaint"] = crop_inpaint
        if inpaint_feather is not None:
            kwargs["inpaint_feather"] = inpaint_feather
        if outscale is not None:
            kwargs["outscale"] = outscale
        if upscale is not None:
            kwargs["upscale"] = upscale
        if upscale_diffusion is not None:
            kwargs["upscale_diffusion"] = upscale_diffusion
        if upscale_iterative is not None:
            kwargs["upscale_iterative"] = upscale_iterative
        if upscale_pipeline is not None:
            kwargs["upscale_pipeline"] = upscale_pipeline
        if upscale_diffusion_steps is not None:
            kwargs["upscale_diffusion_steps"] = upscale_diffusion_steps
        if upscale_diffusion_guidance_scale is not None:
            kwargs["upscale_diffusion_guidance_scale"] = upscale_diffusion_guidance_scale
        if upscale_diffusion_strength is not None:
            kwargs["upscale_diffusion_strength"] = upscale_diffusion_strength
        if upscale_diffusion_prompt is not None:
            kwargs["upscale_diffusion_prompt"] = upscale_diffusion_prompt
        if upscale_diffusion_prompt_2 is not None:
            kwargs["upscale_diffusion_prompt_2"] = upscale_diffusion_prompt_2
        if upscale_diffusion_negative_prompt is not None:
            kwargs["upscale_diffusion_negative_prompt"] = upscale_diffusion_negative_prompt
        if upscale_diffusion_negative_prompt_2 is not None:
            kwargs["upscale_diffusion_negative_prompt_2"] = upscale_diffusion_negative_prompt_2
        if upscale_diffusion_controlnet is not None:
            kwargs["upscale_diffusion_controlnet"] = upscale_diffusion_controlnet
        if upscale_diffusion_chunking_size is not None:
            kwargs["upscale_diffusion_chunking_size"] = upscale_diffusion_chunking_size
        if upscale_diffusion_chunking_blur is not None:
            kwargs["upscale_diffusion_chunking_blur"] = upscale_diffusion_chunking_blur
        if upscale_diffusion_scale_chunking_size is not None:
            kwargs["upscale_diffusion_scale_chunking_size"] = upscale_diffusion_scale_chunking_size
        if upscale_diffusion_scale_chunking_blur is not None:
            kwargs["upscale_diffusion_scale_chunking_blur"] = upscale_diffusion_scale_chunking_blur
        
        logger.info(f"Invoking with keyword arguments {kwargs}")
        
        try:
            response = self.post("/api/invoke", data=kwargs).json()
        except Exception as ex:
            if "responsive" in str(ex).lower():
                logger.warning("Engine process died before becoming responsive, trying one more time.")
                response = self.post("/api/invoke", data=kwargs).json()
            else:
                raise
        return RemoteInvocation.from_response(self, response.get("data", {}))
