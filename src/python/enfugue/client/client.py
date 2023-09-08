from __future__ import annotations

import os
import time

from typing import Dict, Any, Optional, Literal, TypedDict, List, Union
from PIL.Image import Image

from urllib.parse import urlparse

from pibble.api.client.webservice.jsonapi import JSONWebServiceAPIClient
from pibble.ext.user.client.base import UserExtensionClientBase

from enfugue.diffusion.plan import NodeDict, UpscaleStepDict
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
        if "path" not in client:
            client["path"] = "/api"
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
        return self.get("/checkpoints").json()["data"]

    def lora(self) -> List[str]:
        """
        Gets a list of installed lora.
        """
        return self.get("/lora").json()["data"]

    def lycoris(self) -> List[str]:
        """
        Gets a list of installed lycoris.
        """
        return self.get("/lycoris").json()["data"]

    def inversion(self) -> List[str]:
        """
        Gets a list of installed inversion.
        """
        return self.get("/inversion").json()["data"]

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
        status = self.post("/download", data=data).json()["data"]
        while status["status"] != "complete":
            time.sleep(polling_interval)
            statuses = self.get("/download").json()["data"]
            for download_status in statuses:
                if download_status["filename"] == filename:
                    status = download_status
                    break

    def status(self) -> Dict[str, Any]:
        """
        Gets the status from the API.
        """
        return self.get("").json()["data"]

    def settings(self) -> Dict[str, Any]:
        """
        Gets settings from the remote server.
        """
        return self.get("/settings").json()["data"]

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
        chunking_mask_type: Optional[MASK_TYPE_LITERAL] = None,
        chunking_mask_kwargs: Optional[Dict[str, Any]] = None,
        samples: Optional[int] = None,
        iterations: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        refiner_strength: Optional[float] = None,
        refiner_start: Optional[float] = None,
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
        ip_adapter_image: Optional[Union[str, Image]] = None,
        ip_adapter_scale: Optional[float] = None,
        ip_adapter_image_fit: Optional[IMAGE_FIT_LITERAL] = None,
        ip_adapter_image_anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        control_images: Optional[List[Dict[str, Any]]] = None,
        strength: Optional[float] = None,
        fit: Optional[IMAGE_FIT_LITERAL] = None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        remove_background: Optional[bool] = None,
        fill_background: Optional[bool] = None,
        scale_to_model_size: Optional[bool] = None,
        invert_mask: Optional[bool] = None,
        conditioning_scale: Optional[float] = None,
        crop_inpaint: Optional[bool] = None,
        inpaint_feather: Optional[int] = None,
        upscale_steps: Optional[Union[UpscaleStepDict, List[UpscaleStepDict]]] = None,
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
        if chunking_mask_type is not None:
            kwargs["chunking_mask_type"] = chunking_mask_type
        if chunking_mask_kwargs is not None:
            kwargs["chunking_mask_kwargs"] = chunking_mask_kwargs
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
        if refiner_start is not None:
            kwargs["refiner_start"] = refiner_start
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
        if control_images is not None:
            kwargs["control_images"] = control_images
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
        if invert_mask is not None:
            kwargs["invert_mask"] = invert_mask
        if crop_inpaint is not None:
            kwargs["crop_inpaint"] = crop_inpaint
        if inpaint_feather is not None:
            kwargs["inpaint_feather"] = inpaint_feather
        if ip_adapter_scale is not None:
            kwargs["ip_adapter_scale"] = ip_adapter_scale
        if ip_adapter_image is not None:
            kwargs["ip_adapter_image"] = ip_adapter_image
        if ip_adapter_image_fit is not None:
            kwargs["ip_adapter_image_fit"] = ip_adapter_image_fit
        if ip_adapter_image_anchor is not None:
            kwargs["ip_adapter_image_anchor"] = ip_adapter_image_anchor
        if upscale_steps is not None:
            kwargs["upscale_steps"] = upscale_steps

        logger.info(f"Invoking with keyword arguments {kwargs}")

        try:
            response = self.post("/invoke", data=kwargs).json()
        except Exception as ex:
            if "responsive" in str(ex).lower():
                logger.warning("Engine process died before becoming responsive, trying one more time.")
                response = self.post("/invoke", data=kwargs).json()
            else:
                raise
        return RemoteInvocation.from_response(self, response.get("data", {}))
