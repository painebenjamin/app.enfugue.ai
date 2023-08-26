import os
import glob
import PIL
import PIL.Image

from typing import Dict, List, Any, Union, Tuple, Optional
from webob import Request, Response

from pibble.ext.user.server.base import (
    UserExtensionServerBase,
    UserExtensionHandlerRegistry,
    UserExtensionTemplateServer,
)
from pibble.ext.session.server.base import SessionExtensionServerBase
from pibble.ext.rest.server.user import UserRESTExtensionServerBase
from pibble.api.middleware.database.orm import ORMMiddlewareBase
from pibble.api.exceptions import NotFoundError, BadRequestError

from enfugue.diffusion.plan import DiffusionPlan
from enfugue.diffusion.constants import (
    DEFAULT_MODEL,
    DEFAULT_INPAINTING_MODEL,
    DEFAULT_SDXL_MODEL,
    DEFAULT_SDXL_REFINER,
)
from enfugue.api.controller.base import EnfugueAPIControllerBase

__all__ = ["EnfugueAPIInvocationController"]

DEFAULT_MODEL_CKPT = os.path.basename(DEFAULT_MODEL)
DEFAULT_INPAINTING_MODEL_CKPT = os.path.basename(DEFAULT_INPAINTING_MODEL)
DEFAULT_SDXL_MODEL_CKPT = os.path.basename(DEFAULT_SDXL_MODEL)
DEFAULT_SDXL_REFINER_CKPT = os.path.basename(DEFAULT_SDXL_REFINER)

class EnfugueAPIInvocationController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    @property
    def thumbnail_height(self) -> int:
        """
        Gets the height of thumbnails.
        """
        return self.configuration.get("enfugue.thumbnail", 200)

    def get_default_model(self, model: str) -> Optional[str]:
        """
        Gets a default model link by model name, if one exists
        """
        base_model_name = os.path.basename(model)
        if base_model_name == DEFAULT_MODEL_CKPT:
            return DEFAULT_MODEL
        if base_model_name == DEFAULT_INPAINTING_MODEL_CKPT:
            return DEFAULT_INPAINTING_MODEL
        if base_model_name == DEFAULT_SDXL_MODEL_CKPT:
            return DEFAULT_SDXL_MODEL
        if base_model_name == DEFAULT_SDXL_REFINER_CKPT:
            return DEFAULT_SDXL_REFINER
        return None

    def check_find_model(self, model_type: str, model: str) -> str:
        """
        Tries to find a model in a configured directory, if the
        passed model is not an absolute path.
        """
        if os.path.exists(model):
            return model
        check_model = os.path.abspath(
            os.path.join(
                self.configuration.get(
                    f"enfugue.engine.{model_type}", os.path.join(self.engine_root, model_type)
                ),
                model
            )
        )
        if os.path.exists(check_model):
            return check_model
        check_default_model = self.get_default_model(check_model)
        if check_default_model:
            return check_default_model
        raise BadRequestError(f"Cannot find or access {model} (tried {check_model})")

    def check_find_adaptations(
        self,
        model_type: str,
        is_weighted: bool,
        model: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = None,
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Tries to find a model or list of models in a configured directory,
        with or without weights.
        """
        if model is None:
            return []
        elif isinstance(model, str):
            if is_weighted:
                return [(self.check_find_model(model_type, model), 1.0)]
            return [self.check_find_model(model_type, model)]
        elif isinstance(model, dict):
            model_name = model.get("model", None)
            model_weight = model.get("weight", 1.0)
            if not model_name:
                raise BadRequestError(f"Bad model format - missing required dictionary key `model`")
            if is_weighted:
                return [(self.check_find_model(model_type, model_name), model_weight)]
            return [self.check_find_model(model_type, model_name)]
        elif isinstance(model, list):
            models = []
            for item in model:
                models.extend(self.check_find_adaptations(model_type, is_weighted, item))
            return models
        raise BadRequestError(f"Bad format for {model_type} - must be either a single string, a dictionary with the key `model` and optionally `weight`, or a list of the same (got {model})")

    @handlers.path("^/api/invoke$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("DiffusionInvocation", "create")
    def invoke_engine(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        Invokes the engine. Form a plan from the payload, then put the
        invocation in the queue.
        """
        # Get details about model
        model_name = request.parsed.pop("model", None)
        model_type = request.parsed.pop("model_type", None)
        plan_kwargs: Dict[str, Any] = {}

        # Infer type if not passed
        if model_name and not model_type:
            model_type = "checkpoint" if ".ckpt" in model_name or ".safetensors" in model_name else "model"

        if model_name and model_type == "model":
            plan_kwargs = self.get_plan_kwargs_from_model(model_name)
            # Now remove things from the payload that should not be overridable
            for ignored_arg in [
                "lora",
                "lycoris",
                "inversion",
                "vae",
                "size",
                "refiner",
                "refiner_size",
                "refiner_vae",
                "inpainter",
                "inpainter_size",
                "inpainter_vae",
            ]:
                request.parsed.pop(ignored_arg, None)
        elif model_name and model_type == "checkpoint":
            plan_kwargs["model"] = self.check_find_model("checkpoint", model_name)

            refiner = request.parsed.pop("refiner", None)
            plan_kwargs["refiner"] = self.check_find_model("checkpoint", refiner) if refiner else None
            if "refiner" not in plan_kwargs:
                request.parsed.pop("refiner_size", None) # Don't allow override if not overriding checkpoint
                request.parsed.pop("refiner_vae", None)

            inpainter = request.parsed.pop("inpainter", None)
            plan_kwargs["inpainter"] = self.check_find_model("checkpoint", inpainter) if inpainter else None

            if "inpainter" not in plan_kwargs:
                request.parsed.pop("inpainter_size", None)
                request.parsed.pop("inpainter_vae", None)

            lora = request.parsed.pop("lora", [])
            plan_kwargs["lora"] = self.check_find_adaptations("lora", True, lora) if lora else None

            lycoris = request.parsed.pop("lycoris", [])
            plan_kwargs["lycoris"] = self.check_find_adaptations("lycoris", True, lycoris) if lycoris else None

            inversion = request.parsed.pop("inversion", [])
            plan_kwargs["inversion"] = self.check_find_adaptations("inversion", False, inversion) if inversion else None
        # Always take passed scheduler
        scheduler = request.parsed.pop("scheduler", None)
        if scheduler:
            plan_kwargs["scheduler"] = scheduler
        disable_decoding = request.parsed.pop("intermediates", None) == False
        ui_state: Optional[str] = None
        for key, value in request.parsed.items():
            if key == "state":
                ui_state = value
            elif value is not None:
                plan_kwargs[key] = value
        plan = DiffusionPlan.assemble(**plan_kwargs)
        return self.invoke(
            request.token.user.id,
            plan,
            ui_state=ui_state,
            disable_intermediate_decoding=disable_decoding
        ).format()

    @handlers.path("^/api/invocation$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionInvocation", "read")
    def invocations(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Gets all invocations since engine start for a user.
        """
        return [invocation.format() for invocation in self.manager.get_invocations(request.token.user.id)]

    @handlers.path("^/api/invocation/intermediates/(?P<file_path>.+)$")
    @handlers.download()
    @handlers.methods("GET")
    @handlers.compress()
    @handlers.reverse("Intermediate", "/api/invocation/intermediates/{file_path}")
    @handlers.bypass(
        UserRESTExtensionServerBase,
        UserExtensionServerBase,
        ORMMiddlewareBase,
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
    )  # bypass processing for speed
    def download_intermediate_image(self, request: Request, response: Response, file_path: str) -> str:
        """
        Downloads one of the intermediate results of an invocation.
        """
        image_path = os.path.join(self.manager.engine_intermediate_dir, file_path)
        if not os.path.exists(image_path):
            raise NotFoundError(f"No image at {file_path}")
        return image_path

    @handlers.path("^/api/invocation/images/(?P<file_path>.+)$")
    @handlers.download()
    @handlers.methods("GET")
    @handlers.compress()
    @handlers.cache()
    @handlers.reverse("Image", "/api/invocation/images/{file_path}")
    @handlers.bypass(
        UserRESTExtensionServerBase,
        UserExtensionServerBase,
        ORMMiddlewareBase,
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
    )  # bypass processing for speed
    def download_image(self, request: Request, response: Response, file_path: str) -> str:
        """
        Downloads one of the results of an invocation.
        """
        image_path = os.path.join(self.manager.engine_image_dir, file_path)
        if not os.path.exists(image_path):
            raise NotFoundError(f"No image at {file_path}")
        return image_path

    @handlers.path("^/api/invocation/thumbnails/(?P<file_path>.+)$")
    @handlers.download()
    @handlers.methods("GET")
    @handlers.compress()
    @handlers.cache()
    @handlers.reverse("Image", "/api/invocation/thumbnails/{file_path}")
    @handlers.bypass(
        UserRESTExtensionServerBase,
        UserExtensionServerBase,
        ORMMiddlewareBase,
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
    )  # bypass processing for speed
    def download_image_thumbnail(self, request: Request, response: Response, file_path: str) -> str:
        """
        Downloads one of the results of an invocation only as a thumbnail
        """
        image_path = os.path.join(self.manager.engine_image_dir, file_path)
        if not os.path.exists(image_path):
            raise NotFoundError(f"No image at {file_path}")
        image_name, ext = os.path.splitext(os.path.basename(image_path))
        thumbnail_path = os.path.join(self.manager.engine_image_dir, f"{image_name}_thumb{ext}")
        if not os.path.exists(thumbnail_path):
            image = PIL.Image.open(image_path)
            width, height = image.size
            scale = self.thumbnail_height / height
            image.resize((int(width * scale), int(height * scale))).save(thumbnail_path)
        return thumbnail_path

    @handlers.path("^/api/invocation/nsfw$")
    @handlers.methods("GET")
    @handlers.cache()
    @handlers.reverse("NSFW", "/api/invocation/nsfw")
    @handlers.bypass(
        UserRESTExtensionServerBase,
        UserExtensionServerBase,
        ORMMiddlewareBase,
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
    )  # bypass processing for speed
    def download_nsfw_image(self, request: Request, response: Response) -> None:
        """
        The image served when NSFW content was detected
        """
        contents = '<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg" style="background-color: #222222;" width="512" height="512">'
        contents += "<style>text { font: bold 25px sans-serif; fill: #555555; letter-spacing: 1px; } g { transform: rotateZ(-45deg); transform-origin: center; }</style><g>"
        for i in range(40):
            for j in range(24):
                contents += f'<text x="{(j*80)+((i%10)*8)-140}" y="{i*24-80}">NSFW</text>'
        contents += "</g></svg>"
        response.content_type = "image/svg+xml"
        response.text = contents

    @handlers.path("^/api/invocation/(?P<uuid>[a-zA-Z0-9\-]+)$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionInvocation", "read")
    def invocation(self, request: Request, response: Response, uuid: str) -> Dict[str, Any]:
        """
        Gets a single set of invocation details by UUID.
        """
        for invocation in self.manager.get_invocations(request.token.user.id):
            if invocation.uuid == uuid:
                formatted = invocation.format()
                database_invocation = (
                    self.database.query(self.orm.DiffusionInvocation)
                    .filter(self.orm.DiffusionInvocation.id == uuid)
                    .one()
                )
                formatted_images = formatted.get("images", [])
                if isinstance(formatted_images, list):
                    database_invocation.outputs = len(formatted_images)
                database_invocation.duration = formatted.get("duration", 0)
                if "message" in formatted:
                    database_invocation.error = formatted["message"]
                self.database.commit()
                return formatted
        raise NotFoundError(f"No invocation matching UUID {uuid}")

    @handlers.path("^/api/invocation/(?P<uuid>[a-zA-Z0-9\-]+)$")
    @handlers.methods("DELETE")
    @handlers.format()
    @handlers.secured("DiffusionInvocation", "delete")
    def delete_invocation(self, request: Request, response: Response, uuid: str) -> None:
        """
        Deletes an invocation.
        """
        database_invocation = (
            self.database.query(self.orm.DiffusionInvocation)
            .filter(self.orm.DiffusionInvocation.id == uuid)
            .one_or_none()
        )
        if not database_invocation:
            raise NotFoundError(f"No invocation with ID {uuid}")

        for dirname in [self.manager.engine_image_dir, self.manager.engine_intermediate_dir]:
            for invocation_image in glob.glob(f"{uuid}*.png", root_dir=dirname):
                os.remove(os.path.join(dirname, invocation_image))
        self.database.delete(database_invocation)
        self.database.commit()

    @handlers.path("^/api/invocation/stop$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured()
    def stop_engine(self, request: Request, response: Response) -> None:
        """
        Stops the engine and any invocations.
        """
        self.manager.stop_engine()
