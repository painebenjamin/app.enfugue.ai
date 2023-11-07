from __future__ import annotations
import os
import PIL
import signal
import requests
import datetime
import webbrowser

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from webob import Request, Response

from pibble.api.exceptions import NotFoundError, BadRequestError
from pibble.api.server.webservice.template import TemplateServer
from pibble.api.server.webservice.jsonapi import JSONWebServiceAPIServer
from pibble.ext.user.server.base import UserExtensionHandlerRegistry
from pibble.ext.rest.server.user import UserRESTExtensionServerBase
from pibble.ext.user.database import AuthenticationToken, User
from pibble.util.encryption import Password
from pibble.util.helpers import OutputCatcher

from enfugue.diffusion.invocation import LayeredInvocation

from enfugue.database import *
from enfugue.api.controller import *
from enfugue.interface.helpers import *

from enfugue.api.manager import SystemManager
from enfugue.api.invocations import Invocation
from enfugue.api.downloads import Download
from enfugue.api.config import EnfugueConfiguration

from enfugue.util import (
    check_make_directory,
    get_version,
    get_pending_versions,
    get_gpu_status,
    get_local_static_directory,
    find_file_in_directory,
    logger,
)

if TYPE_CHECKING:
    from enfugue.util import GPUStatusDict

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

# Don't confuse people, we can't do OAuth
AuthenticationToken.Hide(["refresh_token", "id"])
User.Hide(["password", "superuser", "password_expires", "verified"])

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
        check_make_directory(root)
        return root

    def get_configured_directory(self, model_type: str) -> str:
        """
        Gets the configured or default directory for a model type
        """
        dirname = self.configuration.get(
            f"enfugue.engine.{model_type}",
            os.path.join(self.engine_root, model_type),
        )
        if dirname.startswith("~"):
            dirname = os.path.expanduser(dirname)
        return os.path.realpath(dirname)

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
                root_user = session.query(self.orm.User).filter(self.orm.User.username == "enfugue").one()
                root_user.password = Password.hash(open(password_file, "r").read())
                for token in root_user.tokens:
                    session.delete(token)
                session.commit()
            os.remove(password_file)

    def configure(self, **configuration: Any) -> None:
        """
        Override configure() to pass in REST configuration.
        """
        super(EnfugueAPIServerBase, self).configure(rest=EnfugueAPIRESTConfiguration, **configuration)

    def on_destroy(self) -> None:
        """
        Stop the manager thread when the server stops.
        """
        if hasattr(self, "manager"):
            logger.debug("Stopping system manager")
            self.manager.stop_monitor()
            self.manager.stop_engine()
            self.manager.stop_interpolator()

    def format_plan(self, plan: LayeredInvocation) -> Dict[str, Any]:
        """
        Formats a plan for inserting into the database
        """
        def get_image_metadata(image: PIL.Image.Image) -> Dict[str, Any]:
            """
            Gets metadata from an image
            """
            width, height = image.size
            metadata = {"width": width, "height": height, "mode": image.mode}
            if hasattr(image, "filename"):
                metadata["filename"] = image.filename
            return metadata

        def replace_images(serialized: Dict[str, Any]) -> Dict[str, Any]:
            """
            Replaces images with a metadata dictionary
            """
            for key, value in serialized.items():
                if isinstance(value, PIL.Image.Image):
                    serialized[key] = get_image_metadata(value)
                elif isinstance(value, dict):
                    serialized[key] = replace_images(value)
                elif isinstance(value, list):
                    serialized[key] = [
                        replace_images(part)
                        if isinstance(part, dict)
                        else get_image_metadata(part)
                        if isinstance(part, PIL.Image.Image)
                        else part
                        for part in value
                    ]
            return serialized

        return replace_images(plan.serialize())

    def get_plan_kwargs_from_model(
        self,
        model_name: str,
        include_prompts: bool = True,
        include_defaults: bool = True
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

        checkpoint_dir = self.get_configured_directory("checkpoint")
        lora_dir = self.get_configured_directory("lora")
        lycoris_dir = self.get_configured_directory("lycoris")
        inversion_dir = self.get_configured_directory("inversion")
        motion_dir = self.get_configured_directory("motion")

        model = find_file_in_directory(checkpoint_dir, diffusion_model.model)
        if not model:
            raise ValueError(f"Could not find {diffusion_model.model} in {checkpoint_dir}")

        refiner = diffusion_model.refiner
        if refiner:
            refiner_model = find_file_in_directory(checkpoint_dir, refiner[0].model)
            if not refiner_model:
                raise ValueError(f"Could not find {refiner[0].model} in {checkpoint_dir}")
            refiner = refiner_model
        else:
            refiner = None

        inpainter = diffusion_model.inpainter
        if inpainter:
            inpainter_model = os.path.join(checkpoint_dir, inpainter[0].model)
            if not inpainter_model:
                raise ValueError(f"Could not find {inpainter[0].model} in {checkpoint_dir}")
            inpainter = inpainter_model
        else:
            inpainter = None

        scheduler = diffusion_model.scheduler
        if scheduler:
            scheduler = scheduler[0].name

        vae = diffusion_model.vae
        if vae:
            vae = diffusion_model.vae[0].name
        else:
            vae = None

        refiner_vae = diffusion_model.refiner_vae
        if refiner_vae:
            refiner_vae = diffusion_model.refiner_vae[0].name
        else:
            refiner_vae = None

        inpainter_vae = diffusion_model.inpainter_vae
        if inpainter_vae:
            inpainter_vae = diffusion_model.inpainter_vae[0].name
        else:
            inpainter_vae = None

        motion_module = diffusion_model.motion_module
        if motion_module:
            motion_module = diffusion_model.motion_module[0].name
        else:
            motion_module = None

        lora = []
        for lora_model in diffusion_model.lora:
            lora_model_path = find_file_in_directory(lora_dir, lora_model.model)
            if not lora_model_path:
                raise ValueError(f"Could not find {lora_model.model} in {lora_dir}")
            lora.append((lora_model_path, float(lora_model.weight)))

        lycoris = []
        for lycoris_model in diffusion_model.lycoris:
            lycoris_model_path = find_file_in_directory(lycoris_dir, lycoris_model.model)
            if not lycoris_model_path:
                raise ValueError(f"Could not find {lycoris_model.model} in {lycoris_dir}")
            lycoris.append((lycoris_model_path, float(lycoris_model.weight)))

        inversion = []
        for inversion_model in diffusion_model.inversion:
            inversion_model_path = find_file_in_directory(inversion_dir, inversion_model.model)
            if not inversion_model_path:
                raise ValueError(f"Could not find {inversion_model.model} in {inversion_dir}")
            inversion.append(inversion_model_path)

        model_prompt = diffusion_model.prompt
        model_negative_prompt = diffusion_model.negative_prompt

        plan_kwargs: Dict[str, Any] = {
            "model": model,
            "refiner": refiner,
            "inpainter": inpainter,
            "lora": lora,
            "lycoris": lycoris,
            "inversion": inversion,
            "scheduler": scheduler,
            "vae": vae,
            "refiner_vae": refiner_vae,
            "inpainter_vae": inpainter_vae,
            "motion_module": motion_module
        }

        model_config = {}
        for default in diffusion_model.config:
            model_config[default.configuration_key] = default.configuration_value

        if include_prompts:
            plan_kwargs["model_prompt"] = model_prompt
            plan_kwargs["model_negative_prompt"] = model_negative_prompt
            plan_kwargs["model_prompt_2"] = model_config.get("prompt_2", None)
            plan_kwargs["model_negative_prompt_2"] = model_config.get("negative_prompt_2", None)
        if include_defaults:
            plan_kwargs = {**plan_kwargs, **model_config}

        return plan_kwargs

    def invoke(
        self,
        user_id: int,
        plan: LayeredInvocation,
        save: bool = True,
        ui_state: Optional[str] = None,
        disable_intermediate_decoding: bool = False,
        video_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> Invocation:
        """
        Invokes the platform and saves any resulting images, returning their paths.
        """
        invocation = self.manager.invoke(
            user_id,
            plan,
            ui_state=ui_state,
            disable_intermediate_decoding=disable_intermediate_decoding,
            video_rate=video_rate,
            video_codec=self.configuration.get("enfugue.video.codec", "avc1"),
            video_format=self.configuration.get("enfugue.video.format", "mp4"),
            **kwargs
        )
        if save:
            self.database.add(
                self.orm.DiffusionInvocation(
                    id=invocation.uuid,
                    user_id=user_id,
                    plan=self.format_plan(plan)
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
        return self.manager.download(user_id, url, destination, headers=headers, parameters=parameters)

    def __del__(self) -> None:
        """
        Add a __del__ in case this gets deleted before on_destroy is called
        """
        self.on_destroy()

    def get_gpu_status(self) -> Optional[GPUStatusDict]:
        """
        Gets the GPU status, optionally hiding details
        """
        status = get_gpu_status()
        if status is None:
            return None
        for key in list(status.keys()):
            if self.configuration.get(f"enfugue.gpu.{key}", True) in [False, 0, "0"]: # must be falsey, not null/none
                status.pop(key) # type: ignore[misc]
        return status

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
            "gpu": self.get_gpu_status(),
            "version": get_version(),
            "uptime": (datetime.datetime.now() - self.start_time).total_seconds(),
        }

    @handlers.path("^/api/login$")
    @handlers.methods("POST")
    @handlers.format()
    def api_login(self, request: Request, response: Response) -> AuthenticationToken:
        """
        Logs in through the API (and returns the token in JSON)
        """
        return self.login(request, response)

    @handlers.path("^/api/announcements$")
    @handlers.methods("GET")
    @handlers.format()
    @handlers.secured()
    def announcements(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Gets any announcements to display to the user.
        """
        announcements = []

        current_version = get_version()
        snooze_time = self.user_config.get("enfugue.snooze.time", None)
        snooze_version = self.user_config.get("enfugue.snooze.version", None)
        snooze_duration = float("inf")
        if snooze_time is not None:
            snooze_duration = (datetime.datetime.now() - snooze_time).total_seconds()

        is_snoozed = snooze_duration < (60 * 60 * 24 * 30) and snooze_version != current_version

        if not is_snoozed:
            is_initialized = self.user_config.get("enfugue.initialized", False)
            if not is_initialized:
                directories = {}
                for dirname in [
                    "cache",
                    "checkpoint",
                    "diffusers",
                    "lora",
                    "lycoris",
                    "inversion",
                    "other",
                    "tensorrt",
                    "images",
                    "intermediate",
                ]:
                    directories[dirname] = self.configuration.get(
                        f"enfugue.engine.{dirname}", os.path.join(self.engine_root, dirname)
                    )

                announcements.append({"type": "initialize", "directories": directories})

            pending_default_downloads = self.manager.pending_default_downloads
            pending_xl_downloads = self.manager.pending_xl_downloads

            for typename, pending_downloads in [
                ("download", pending_default_downloads),
                ("optional-download", pending_xl_downloads)
            ]:
                for url, dest in pending_downloads:
                    model = os.path.basename(url)
                    announcements.append(
                        {
                            "type": typename,
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
        self.user_config["enfugue.snooze.time"] = datetime.datetime.now()
        self.user_config["enfugue.snooze.version"] = get_version()

    @classmethod
    def serve_icon(cls, configuration: Dict[str, Any]) -> None:
        """
        When working on windows using a .exe, this runs the server using a system tray icon
        """
        import pystray
        from threading import Event

        stop_event = Event()
        catcher = OutputCatcher()

        def stop(icon: pystray.Icon) -> None:
            """
            Stops the server.
            """
            icon.visible = False
            stop_event.set()
            icon.stop()

        def open_app(icon: pystray.Icon) -> None:
            """
            Opens up a browser and navigates to the app
            """
            server_config = configuration.get("server", {})
            scheme = "https" if server_config.get("secure", False) else "http"
            domain = server_config.get("domain", "127.0.0.1")
            port = server_config.get("port", 45554)
            url = f"{scheme}://{domain}:{port}/"
            webbrowser.open(url)

        def setup(icon: pystray.Icon) -> None:
            """
            Starts the server.
            """
            icon.visible = True
            server = cls()
            server.configure_start(signal=False, **configuration)
            while not stop_event.is_set():
                out, err = catcher.output()
                if out:
                    logger.debug(f"STDOUT: {out}")
                if err:
                    logger.debug(f"STDERR: {err}")
                stop_event.wait(1)
            server.stop()
            server.destroy()

        static_dir = get_local_static_directory()
        icon_path = configuration.get("enfugue", {}).get("icon", "favicon/favicon-64x64.png")
        icon_image = PIL.Image.open(os.path.join(static_dir, "img", icon_path))
        icon = pystray.Icon("enfugue", icon_image)
        icon.menu = pystray.Menu(pystray.MenuItem("Open App", open_app), pystray.MenuItem("Quit", stop))
        with catcher:
            icon.run(setup=setup)

server_parents = tuple([EnfugueAPIServerBase] + list(EnfugueAPIControllerBase.enumerate()))

EnfugueAPIServer = type("EnfugueAPIServer", server_parents, {})
