import os
import glob
import PIL
import PIL.Image

from typing import Dict, List, Any
from webob import Request, Response

from pibble.ext.user.server.base import (
    UserExtensionServerBase,
    UserExtensionHandlerRegistry,
    UserExtensionTemplateServer,
)
from pibble.ext.session.server.base import SessionExtensionServerBase
from pibble.ext.rest.server.user import UserRESTExtensionServerBase
from pibble.api.middleware.database.orm import ORMMiddlewareBase
from pibble.api.exceptions import NotFoundError

from enfugue.diffusion.plan import DiffusionPlan
from enfugue.api.controller.base import EnfugueAPIControllerBase

__all__ = ["EnfugueAPIInvocationController"]


class EnfugueAPIInvocationController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    @property
    def thumbnail_height(self) -> int:
        """
        Gets the height of thumbnails.
        """
        return self.configuration.get("enfugue.thumbnail", 200)

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
        model, size, lora, inversion, model_prompt, model_negative_prompt = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        model_name = request.parsed.pop("model", None)
        plan_kwargs: Dict[str, Any] = {}
        if model_name is not None:
            plan_kwargs = self.get_plan_kwargs_from_model(model_name)

        plan = DiffusionPlan.from_nodes(**{**plan_kwargs, **request.parsed})
        return self.invoke(request.token.user.id, plan).format()

    @handlers.path("^/api/invocation$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured("DiffusionInvocation", "read")
    def invocations(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Gets all invocations since engine start for a user.
        """
        return [
            invocation.format()
            for invocation in self.manager.get_invocations(request.token.user.id)
        ]

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
    def download_intermediate_image(
        self, request: Request, response: Response, file_path: str
    ) -> str:
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
