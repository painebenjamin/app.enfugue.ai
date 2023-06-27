import os
import PIL
import signal
import requests
import datetime

from typing import Any, Dict, List

from webob import Request, Response

from pibble.api.exceptions import NotFoundError
from pibble.api.server.webservice.template import TemplateServer
from pibble.api.server.webservice.jsonapi import JSONWebServiceAPIServer
from pibble.ext.user.server.base import UserExtensionHandlerRegistry
from pibble.ext.rest.server.user import UserRESTExtensionServerBase
from pibble.util.encryption import Password

from enfugue.diffusion.plan import DiffusionPlan

from enfugue.database import *
from enfugue.api.controller import *
from enfugue.interface.helpers import *

from enfugue.api.manager import SystemManager
from enfugue.api.invocations import Invocation
from enfugue.api.downloads import Download
from enfugue.api.config import EnfugueConfiguration

from enfugue.util import (
    get_version,
    get_pending_versions,
    get_gpu_status,
    get_local_static_directory,
    logger,
)

# Auto-register some REST handlers
EnfugueAPIRESTConfiguration = {
    "root": "/api",
    "scopes": [
        {"class": "DiffusionModel", "scope": "name", "root": "models", "methods": ["GET"]},
        {"class": "User", "scope": "username", "root": "users", "methods": ["GET"]},
        {
            "class": "DiffusionInvocation",
            "scope": "id",
            "root": "invocation-history",
            "methods": ["GET"],
            "user": {"id": "user_id"},
        },
    ],
}

__all__ = ["EnfugueAPIServerBase", "EnfugueAPIServer"]


class EnfugueAPIServerBase(
    JSONWebServiceAPIServer,
    UserRESTExtensionServerBase,
):
    handlers = UserExtensionHandlerRegistry()

    @property
    def engine_root(self) -> str:
        """
        Returns the engine root location.
        """
        root = self.configuration.get("enfugue.engine.root", "~/.cache/enfugue")
        if root.startswith("~"):
            root = os.path.expanduser(root)
        root = os.path.realpath(root)
        if not os.path.exists(root):
            os.makedirs(root)
        return root

    def on_configure(self) -> None:
        """
        On configuration, we add all the API objects to the ORM, create the template loader,
        the job dispatcher, and make sure we can upload files later.
        """
        self.orm.extend_base(EnfugueObjectBase)
        self.user_config = EnfugueConfiguration(self.orm)
        self.configuration.update(**self.user_config.dict())
        self.manager = SystemManager(self.configuration)
        self.manager.start_monitor()
        self.start_time = datetime.datetime.now()
        self.check_emergency_password_reset()
        if self.configuration.get("signal", True):
            self.register_signal_handlers()

    def register_signal_handlers(self) -> None:
        """
        Catch INT to make sure we destroy.
        """
        original_handler = signal.getsignal(signal.SIGINT)

        def signal_handler(*args: Any) -> None:
            self.on_destroy()
            signal.signal(signal.SIGINT, original_handler)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, signal_handler)

    def check_emergency_password_reset(self) -> None:
        """
        Checks if there is an emergency password reset present in the engine directory.
        """
        password_file = os.path.join(self.engine_root, "password_reset.txt")
        if os.path.exists(password_file):
            with self.orm.session() as session:
                root_user = (
                    session.query(self.orm.User).filter(self.orm.User.username == "enfugue").one()
                )
                root_user.password = Password.hash(open(password_file, "r").read())
                for token in root_user.tokens:
                    session.delete(token)
                session.commit()
            os.remove(password_file)

    def configure(self, **configuration: Any) -> None:
        """
        Override configure() to pass in REST configuration.
        """
        super(EnfugueAPIServerBase, self).configure(
            rest=EnfugueAPIRESTConfiguration, **configuration
        )

    def on_destroy(self) -> None:
        """
        Stop the manager thread when the server stops.
        """
        if hasattr(self, "manager"):
            logger.debug("Stopping system manager")
            self.manager.stop_monitor()
            self.manager.stop_engine()

    def format_plan(self, plan: DiffusionPlan) -> Dict[str, Any]:
        """
        Formats a plan for inserting into the database
        """

        def replace_images(serialized: Dict[str, Any]) -> Dict[str, Any]:
            """
            Replaces images with a metadata dictionary
            """
            for key, value in serialized.items():
                if isinstance(value, PIL.Image.Image):
                    width, height = value.size
                    metadata = {"width": width, "height": height, "mode": value.mode}
                    if hasattr(value, "filename"):
                        metadata["filename"] = value.filename
                    serialized[key] = metadata
                elif isinstance(value, dict):
                    serialized[key] = replace_images(value)
                elif isinstance(value, list):
                    serialized[key] = [
                        replace_images(part) if isinstance(part, dict) else part for part in value
                    ]
            return serialized

        return replace_images(plan.get_serialization_dict())

    def get_plan_kwargs_from_model(
        self, model_name: str, include_prompts: bool = True
    ) -> Dict[str, Any]:
        """
        Given a model name, return the keyword arguments that should be passed into a plan.
        """
        diffusion_model = (
            self.database.query(self.orm.DiffusionModel)
            .filter(self.orm.DiffusionModel.name == model_name)
            .one_or_none()
        )
        if not diffusion_model:
            raise NotFoundError(f"Unknown diffusion model {model_name}")

        model = os.path.join(self.manager.engine_root_dir, "checkpoint", diffusion_model.model)
        size = diffusion_model.size
        lora = [
            (os.path.join(self.manager.engine_root_dir, "lora", lora.model), float(lora.weight))
            for lora in diffusion_model.lora
        ]
        inversion = [
            os.path.join(self.manager.engine_root_dir, "inversion", inversion.model)
            for inversion in diffusion_model.inversion
        ]
        model_prompt = diffusion_model.prompt
        model_negative_prompt = diffusion_model.negative_prompt

        plan_kwargs: Dict[str, Any] = {
            "model": model,
            "size": size,
            "lora": lora,
            "inversion": inversion,
        }

        if include_prompts:
            plan_kwargs["model_prompt"] = model_prompt
            plan_kwargs["model_negative_prompt"] = model_negative_prompt

        return plan_kwargs

    def invoke(
        self,
        user_id: int,
        plan: DiffusionPlan,
        save: bool = True,
        disable_intermediate_decoding: bool = False,
        **kwargs: Any,
    ) -> Invocation:
        """
        Invokes the platform and saves any resulting images, returning their paths.
        """
        invocation = self.manager.invoke(
            user_id, plan, disable_intermediate_decoding=disable_intermediate_decoding, **kwargs
        )
        if save:
            self.database.add(
                self.orm.DiffusionInvocation(
                    id=invocation.uuid, user_id=user_id, plan=self.format_plan(plan)
                )
            )
            self.database.commit()
        return invocation

    def download(
        self,
        user_id: int,
        url: str,
        destination: str,
        headers: Dict[str, str] = {},
        parameters: Dict[str, Any] = {},
    ) -> Download:
        """
        Starts a download using the download manager.
        """
        return self.manager.download(
            user_id, url, destination, headers=headers, parameters=parameters
        )

    def __del__(self) -> None:
        """
        Add a __del__ in case this gets deleted before on_destroy is called
        """
        self.on_destroy()

    @handlers.path("^/api(/?)$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def status(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        Gets the status of the engine and the time since the server was started.
        """
        return {
            "status": self.manager.engine_status,
            "system": self.manager.status,
            "gpu": get_gpu_status(),
            "version": get_version(),
            "uptime": (datetime.datetime.now() - self.start_time).total_seconds(),
        }

    @handlers.path("^/api/announcements$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def announcements(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Gets any announcements to display to the user.
        """
        announcements = []

        snooze_time = self.user_config.get("enfugue.snooze", None)
        snooze_duration = float("inf")
        if snooze_time is not None:
            snooze_duration = (datetime.datetime.now() - snooze_time).total_seconds()
        is_snoozed = snooze_duration < (60 * 60 * 24)
        if not is_snoozed:
            is_initialized = self.user_config.get("enfugue.initialized", False)
            if not is_initialized:
                announcements.append({"type": "initialize"})

            pending_downloads = self.manager.pending_default_downloads
            for url, dest in pending_downloads:
                model = os.path.basename(url)
                announcements.append(
                    {
                        "type": "download",
                        "url": url,
                        "size": requests.head(url, allow_redirects=True).headers["Content-Length"],
                        "model": model,
                        "destination": dest,
                    }
                )
            pending_versions = get_pending_versions()
            for version in pending_versions:
                announcements.append(
                    {
                        "type": "update",
                        "version": version["version"],
                        "release": version["release"].strftime("%Y-%m-%d"),
                        "description": version["description"],
                    }
                )

        return announcements

    @handlers.path("^/api/announcements/snooze$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured()
    def snooze_announcements(self, request: Request, response: Response) -> None:
        """
        Snoozes announcements for a day.
        """
        self.user_config["enfugue.initialized"] = True
        self.user_config["enfugue.snooze"] = datetime.datetime.now()

    @classmethod
    def serve_icon(cls, configuration: Dict[str, Any]) -> None:
        """
        When working on windows using a .exe, this runs the server using a system tray icon
        """
        import pystray
        from threading import Event

        stop_event = Event()

        def stop(icon: pystray.Icon) -> None:
            """
            Stops the server.
            """
            icon.visible = False
            stop_event.set()
            icon.stop()

        def setup(icon: pystray.Icon) -> None:
            """
            Starts the server.
            """
            icon.visible = True
            server = cls()
            server.configure_start(signal=False, **configuration)
            while not stop_event.is_set():
                stop_event.wait(1)
            server.stop()
            server.destroy()

        static_dir = get_local_static_directory()
        icon_path = configuration.get("enfugue", {}).get("icon", "favicon/favicon-64x64.png")
        icon_image = PIL.Image.open(os.path.join(static_dir, "img", icon_path))
        icon = pystray.Icon("enfugue", icon_image)
        icon.menu = pystray.Menu(pystray.MenuItem("Quit", stop))
        icon.run(setup=setup)


server_parents = tuple([EnfugueAPIServerBase] + list(EnfugueAPIControllerBase.enumerate()))

EnfugueAPIServer = type("EnfugueAPIServer", server_parents, {})
