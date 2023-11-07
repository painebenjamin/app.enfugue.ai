from __future__ import annotations

import os
import glob
import PIL
import PIL.Image

from typing import Dict, List, Any, Union, Tuple, Optional, TYPE_CHECKING

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

from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.diffusion.constants import *
from enfugue.util import find_file_in_directory
from enfugue.api.controller.base import EnfugueAPIControllerBase

if TYPE_CHECKING:
    import cv2

__all__ = ["EnfugueAPIInvocationController"]

DEFAULT_MODEL_CKPT = os.path.basename(DEFAULT_MODEL)
DEFAULT_INPAINTING_MODEL_CKPT = os.path.basename(DEFAULT_INPAINTING_MODEL)
DEFAULT_SDXL_MODEL_CKPT = os.path.basename(DEFAULT_SDXL_MODEL)
DEFAULT_SDXL_REFINER_CKPT = os.path.basename(DEFAULT_SDXL_REFINER)
DEFAULT_SDXL_INPAINTING_CKPT = os.path.basename(DEFAULT_SDXL_INPAINTING_MODEL)
MOTION_LORA_ZOOM_OUT_CKPT = os.path.basename(MOTION_LORA_ZOOM_OUT)
MOTION_LORA_ZOOM_IN_CKPT = os.path.basename(MOTION_LORA_ZOOM_IN)
MOTION_LORA_PAN_LEFT_CKPT = os.path.basename(MOTION_LORA_PAN_LEFT)
MOTION_LORA_PAN_RIGHT_CKPT = os.path.basename(MOTION_LORA_PAN_RIGHT)
MOTION_LORA_TILT_UP_CKPT = os.path.basename(MOTION_LORA_TILT_UP)
MOTION_LORA_TILT_DOWN_CKPT = os.path.basename(MOTION_LORA_TILT_DOWN)
MOTION_LORA_ROLL_CLOCKWISE_CKPT = os.path.basename(MOTION_LORA_ROLL_CLOCKWISE)
MOTION_LORA_ROLL_ANTI_CLOCKWISE_CKPT = os.path.basename(MOTION_LORA_ROLL_ANTI_CLOCKWISE)

class EnfugueAPIInvocationController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    @property
    def thumbnail_height(self) -> int:
        """
        Gets the height of thumbnails.
        """
        return self.configuration.get("enfugue.thumbnail", 100)

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
        if base_model_name == DEFAULT_SDXL_INPAINTING_CKPT:
            return DEFAULT_SDXL_INPAINTING_MODEL
        if base_model_name == MOTION_LORA_ZOOM_OUT_CKPT:
            return MOTION_LORA_ZOOM_OUT
        if base_model_name == MOTION_LORA_ZOOM_IN_CKPT:
            return MOTION_LORA_ZOOM_IN
        if base_model_name == MOTION_LORA_PAN_LEFT_CKPT:
            return MOTION_LORA_PAN_LEFT
        if base_model_name == MOTION_LORA_PAN_RIGHT_CKPT:
            return MOTION_LORA_PAN_RIGHT
        if base_model_name == MOTION_LORA_TILT_UP_CKPT:
            return MOTION_LORA_TILT_UP
        if base_model_name == MOTION_LORA_TILT_DOWN_CKPT:
            return MOTION_LORA_TILT_DOWN
        if base_model_name == MOTION_LORA_ROLL_CLOCKWISE_CKPT:
            return MOTION_LORA_ROLL_CLOCKWISE
        if base_model_name == MOTION_LORA_ROLL_ANTI_CLOCKWISE_CKPT:
            return MOTION_LORA_ROLL_ANTI_CLOCKWISE
        return None

    def get_default_size_for_model(self, model: Optional[str]) -> int:
        """
        Gets the default size for the model.
        """
        if model is None:
            model = self.configuration.get("enfugue.model", DEFAULT_MODEL)
        model_name = os.path.splitext(os.path.basename(model))[0]
        diffusers_path = os.path.join(
            self.configuration.get("enfugue.engine.diffusers", "~/.cache/enfugue/diffusers"),
            model_name
        )
        if diffusers_path.startswith("~"):
            diffusers_path = os.path.expanduser(diffusers_path)
        if os.path.exists(diffusers_path) and os.path.exists(os.path.join(diffusers_path, "text_encoder_2")):
            return 1024
        return 1024 if "xl" in model_name.lower() else 512

    def check_find_model(self, model_type: str, model: str) -> str:
        """
        Tries to find a model in a configured directory, if the
        passed model is not an absolute path.
        """
        if os.path.exists(model):
            return model
        model_basename = os.path.splitext(os.path.basename(model))[0]
        model_dir = self.get_configured_directory(model_type)
        existing_model = find_file_in_directory(
            model_dir,
            model_basename,
            extensions = [".ckpt", ".bin", ".pt", ".pth", ".safetensors"]
        )
        if existing_model:
            return existing_model
        check_default_model = self.get_default_model(model)
        if check_default_model:
            return check_default_model
        raise BadRequestError(f"Cannot find or access {model} (looked recursively for AI model checkpoint formats named {model_basename} in {model_dir})")

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
                return []
            if is_weighted:
                return [(self.check_find_model(model_type, model_name), model_weight)]
            return [self.check_find_model(model_type, model_name)]
        elif isinstance(model, list):
            models = []
            for item in model:
                models.extend(
                    self.check_find_adaptations(model_type, is_weighted, item)
                )
            return models
        raise BadRequestError(f"Bad format for {model_type} - must be either a single string, a dictionary with the key `model` and optionally `weight`, or a list of the same (got {model})")

    def convert_animation(
        self,
        source_path: str,
        dest_path: str,
        rate: float,
    ) -> str:
        """
        Converts animation file formats
        """
        def on_open(capture: cv2.VideoCapture) -> None:
            nonlocal rate
            import cv2
            rate = capture.get(cv2.CAP_PROP_FPS)

        from enfugue.diffusion.util import Video
        frames = [
            frame for frame in 
            Video.file_to_frames(
                source_path,
                on_open=on_open,
            )
        ] # Memoize so we capture rate
        Video(frames).save(
            dest_path,
            rate=rate,
            overwrite=True
        )

        return dest_path

    def get_animation(
        self,
        file_path: str,
        rate: float=8.0,
        overwrite: bool=False,
    ) -> str:
        """
        Gets an animation
        """
        video_path = os.path.join(self.manager.engine_image_dir, file_path)
        base, ext = os.path.splitext(video_path)
        if not os.path.exists(video_path) or overwrite:
            if ext != ".mp4":
                # Look for mp4
                mp4_path = f"{base}.mp4"
                if os.path.exists(mp4_path):
                    return self.convert_animation(mp4_path, video_path, rate)

            from enfugue.diffusion.util import Video

            images = []
            image_id, _ = os.path.splitext(os.path.basename(video_path))
            frame = 0

            while True:
                image_path = os.path.join(self.manager.engine_image_dir, f"{image_id}_{frame}.png")
                if not os.path.exists(image_path):
                    break
                images.append(image_path)
                frame += 1

            if not images:
                raise NotFoundError(f"No images for ID {image_id}")

            frames = [
                PIL.Image.open(image) for image in images
            ]

            Video(frames).save(
                video_path,
                rate=rate,
                overwrite=True
            )

        return video_path

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
                "motion_module",
            ]:
                request.parsed.pop(ignored_arg, None)

        elif model_name and model_type in ["checkpoint", "diffusers", "checkpoint+diffusers"]:
            if model_type == "diffusers":
                plan_kwargs["model"] = model_name # Hope for the best
            else:
                plan_kwargs["model"] = self.check_find_model("checkpoint", model_name)

            refiner = request.parsed.pop("refiner", None)
            if refiner is not None:
                if "." in refiner:
                    plan_kwargs["refiner"] = self.check_find_model("checkpoint", refiner)
                else:
                    plan_kwargs["refiner"] = refiner

            if "refiner" not in plan_kwargs:
                request.parsed.pop("refiner_size", None) # Don't allow override if not overriding checkpoint
                request.parsed.pop("refiner_vae", None)

            inpainter = request.parsed.pop("inpainter", None)
            if inpainter is not None:
                if "." in inpainter:
                    plan_kwargs["inpainter"] = self.check_find_model("checkpoint", inpainter)
                else:
                    plan_kwargs["inpainter"] = inpainter

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
        video_rate: Optional[float] = None

        for key, value in request.parsed.items():
            if key == "state":
                ui_state = value
            elif key == "frame_rate":
                video_rate = value
            elif value is not None:
                plan_kwargs[key] = value

        if not plan_kwargs.get("size", None):
            plan_kwargs["size"] = self.get_default_size_for_model(
                plan_kwargs.get("model", None)
            )

        plan = LayeredInvocation.assemble(**plan_kwargs)

        return self.invoke(
            request.token.user.id,
            plan,
            ui_state=ui_state,
            video_rate=video_rate,
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

    @handlers.path("^/api/invocation/animation/images/(?P<file_path>.+)$")
    @handlers.download()
    @handlers.methods("GET")
    @handlers.compress()
    @handlers.cache()
    @handlers.reverse("Animation", "/api/invocation/animation/images/{file_path}")
    @handlers.bypass(
        UserRESTExtensionServerBase,
        UserExtensionServerBase,
        ORMMiddlewareBase,
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
    )  # bypass processing for speed
    def download_animation(self, request: Request, response: Response, file_path: str) -> str:
        """
        Downloads all results of an invocation as a video
        """
        video_path = os.path.join(self.manager.engine_image_dir, file_path)
        try:
            rate = float(request.params.get("rate", 8.0))
        except:
            rate = 8.0
        return self.get_animation(
            file_path,
            rate=rate,
            overwrite=bool(request.params.get("overwrite", 0))
        )

    @handlers.path("^/api/invocation/thumbnails/(?P<file_path>.+)$")
    @handlers.download()
    @handlers.methods("GET")
    @handlers.compress()
    @handlers.cache()
    @handlers.reverse("Thumbnail", "/api/invocation/thumbnails/{file_path}")
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

    @handlers.path("^/api/invocation/animation/thumbnails/(?P<file_path>.+)$")
    @handlers.download()
    @handlers.methods("GET")
    @handlers.compress()
    @handlers.cache()
    @handlers.reverse("AnimationThumbnail", "/api/invocation/animation/thumbnails/{file_path}")
    @handlers.bypass(
        UserRESTExtensionServerBase,
        UserExtensionServerBase,
        ORMMiddlewareBase,
        SessionExtensionServerBase,
        UserExtensionTemplateServer,
    )  # bypass processing for speed
    def download_animation_thumbnail(self, request: Request, response: Response, file_path: str) -> str:
        """
        Downloads all results of an invocation as a thumbnail video
        """
        image_name, ext = os.path.splitext(os.path.basename(file_path))
        thumbnail_path = os.path.join(self.manager.engine_image_dir, f"{image_name}_thumb{ext}")

        if not os.path.exists(thumbnail_path):
            try:
                rate = float(request.params.get("rate", 8.0))
            except:
                rate = 8.0

            video_path = self.get_animation(
                file_path,
                rate=rate,
            )

            def on_open(capture: cv2.VideoCapture) -> None:
                nonlocal rate
                import cv2
                rate = capture.get(cv2.CAP_PROP_FPS)

            from enfugue.diffusion.util import Video
            frames = [
                frame for frame in 
                Video.file_to_frames(
                    video_path,
                    on_open=on_open,
                    resolution=self.thumbnail_height
                )
            ] # Memoize so we capture rate
            Video(frames).save(
                thumbnail_path,
                rate=rate,
                overwrite=True
            )

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
            for invocation_image in glob.glob(f"{uuid}*.*", root_dir=dirname):
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
        self.manager.stop_interpolator()
