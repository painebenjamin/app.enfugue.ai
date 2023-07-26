from __future__ import annotations

import os
import time

from typing import Dict, Any, Optional, Literal, TypedDict, List, Union

from urllib.parse import urlparse

from pibble.api.client.webservice.jsonapi import JSONWebServiceAPIClient
from pibble.ext.user.client.base import UserExtensionClientBase

from enfugue.diffusion.plan import NodeDict
from enfugue.diffusion.constants import (
    SCHEDULER_LITERAL,
    MULTI_SCHEDULER_LITERAL,
    CONTROLNET_LITERAL,
    UPSCALE_LITERAL,
    VAE_LITERAL,
)
from enfugue.util import logger
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
        negative_prompt: Optional[str] = None,
        model_prompt: Optional[str] = None,
        model_negative_prompt: Optional[str] = None,
        intermediates: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        chunking_size: Optional[int] = None,
        chunking_blur: Optional[int] = None,
        samples: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        refiner_strength: Optional[float] = None,
        refiner_guidance_scale: Optional[float] = None,
        refiner_aesthetic_score: Optional[float] = None,
        refiner_negative_aesthetic_score: Optional[float] = None,
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
        inversion: Optional[str] = None,
        scheduler: Optional[SCHEDULER_LITERAL] = None,
        multi_scheduler: Optional[MULTI_SCHEDULER_LITERAL] = None,
        vae: Optional[VAE_LITERAL] = None,
        seed: Optional[int] = None,
        outscale: Optional[int] = 1,
        upscale: Optional[Union[UPSCALE_LITERAL, List[UPSCALE_LITERAL]]] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = None,
        upscale_diffusion_guidance_scale: Optional[Union[Union[int, float], List[Union[int, float]]]] = None,
        upscale_diffusion_strength: Optional[Union[float, List[float]]] = None,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_negative_prompt: Optional[Union[str, List[str]]] = None,
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

        if prompt is not None:
            kwargs["prompt"] = prompt
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if model_prompt is not None:
            kwargs["model_prompt"] = model_prompt
        if model_negative_prompt is not None:
            kwargs["model_negative_prompt"] = model_negative_prompt
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
        if nodes is not None:
            kwargs["nodes"] = nodes
        if model is not None:
            kwargs["model"] = model
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
        if multi_scheduler is not None:
            kwargs["multi_scheduler"] = multi_scheduler
        if vae is not None:
            kwargs["vae"] = vae
        if seed is not None:
            kwargs["seed"] = seed
        if outscale is not None:
            kwargs["outscale"] = outscale
        if upscale is not None:
            kwargs["upscale"] = upscale
        if upscale_diffusion is not None:
            kwargs["upscale_diffusion"] = upscale_diffusion
        if upscale_iterative is not None:
            kwargs["upscale_iterative"] = upscale_iterative
        if upscale_diffusion_steps is not None:
            kwargs["upscale_diffusion_steps"] = upscale_diffusion_steps
        if upscale_diffusion_guidance_scale is not None:
            kwargs["upscale_diffusion_guidance_scale"] = upscale_diffusion_guidance_scale
        if upscale_diffusion_strength is not None:
            kwargs["upscale_diffusion_strength"] = upscale_diffusion_strength
        if upscale_diffusion_prompt is not None:
            kwargs["upscale_diffusion_prompt"] = upscale_diffusion_prompt
        if upscale_diffusion_negative_prompt is not None:
            kwargs["upscale_diffusion_negative_prompt"] = upscale_diffusion_negative_prompt
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
        if model_type is not None:
            kwargs["model_type"] = model_type
        elif model is not None:
            kwargs["model_type"] = "checkpoint" if "." in model else "model"
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
