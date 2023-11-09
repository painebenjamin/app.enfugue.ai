from __future__ import annotations
import os
import time
import datetime

from typing import Optional, Dict, List, Any, Tuple, TypedDict
from threading import Thread, Event
from multiprocessing import Lock

from pibble.api.configuration import APIConfiguration
from pibble.api.exceptions import TooManyRequestsError, BadRequestError
from pibble.util.numeric import human_size

from enfugue.api.downloads import Download
from enfugue.api.invocations import Invocation
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.diffusion.interpolate import InterpolationEngine

from enfugue.diffusion.invocation import LayeredInvocation
from enfugue.util import logger, check_make_directory, find_file_in_directory
from enfugue.diffusion.constants import (
    DEFAULT_MODEL,
    DEFAULT_INPAINTING_MODEL,
    DEFAULT_SDXL_MODEL,
    DEFAULT_SDXL_REFINER,
    DEFAULT_SDXL_INPAINTING_MODEL,
)

__all__ = ["SystemManagerThread", "SystemManager"]


class SystemManagerThread(Thread):
    """
    This thread simply executes periodic tasks on the manager.
    """

    def __init__(self, manager: SystemManager) -> None:
        super(SystemManagerThread, self).__init__()
        self.manager = manager
        self.stop_event = Event()

    def stop(self) -> None:
        """
        Set the stop event.
        """
        self.stop_event.set()

    @property
    def stopped(self) -> bool:
        """
        Returns true if the stop event is set.
        """
        return self.stop_event.is_set()

    def run(self) -> None:
        """
        The main thread loop.
        """
        while not self.stopped:
            if not self.manager.running:
                logger.info("Manager exited, stopping manager thread.")
                return
            self.manager.do_periodic_tasks()
            time.sleep(0.5)


class SystemManagerDownloadStatusDict(TypedDict):
    active: int
    queued: int
    total: int


class SystemManagerInvocationStatusDict(TypedDict):
    active: bool
    queued: int
    total: int


class SystemManagerStatusDict(TypedDict):
    downloads: SystemManagerDownloadStatusDict
    invocations: SystemManagerInvocationStatusDict


class SystemManager:
    """
    This single class serves to perform all necessary asynchronous processes
    for the API, to reduce headache in wrangling many queues and processes.
    """

    DEFAULT_MAX_CONCURRENT_DOWNLOADS = 2
    DEFAULT_MAX_QUEUED_DOWNLOADS = 10
    DEFAULT_MAX_QUEUED_INVOCATIONS = 2

    downloads: Dict[int, List[Download]]
    invocations: Dict[int, List[Invocation]]
    download_queue: List[Download]
    invocation_queue: List[Invocation]
    active_invocation: Optional[Invocation]

    def __init__(self, configuration: APIConfiguration) -> None:
        self.lock = Lock()
        self.active_invocation = None
        self.configuration = configuration
        self.engine = DiffusionEngine(self.configuration)
        self.interpolator = InterpolationEngine(self.configuration)
        self.downloads = {}
        self.invocations = {}
        self.download_queue = []
        self.invocation_queue = []
        self.thread = SystemManagerThread(self)
        self.running = False

    def start_monitor(self) -> None:
        """
        Starts the system manager thread.
        """
        logger.debug("Starting system monitor")
        self.running = True
        self.thread.start()

    def stop_monitor(self) -> None:
        """
        Stops the system manager thread.
        """
        logger.debug("Stopping system monitor")
        self.running = False
        self.thread.stop()
        self.thread.join()

    @property
    def engine_root_dir(self) -> str:
        """
        Gets the root location for the engine.
        """
        root = self.configuration.get("enfugue.engine.root", "~/.cache/enfugue")
        if root.startswith("~"):
            root = os.path.expanduser(root)
        root = os.path.realpath(root)
        check_make_directory(root)
        return root

    @property
    def engine_image_dir(self) -> str:
        """
        Gets the location for image result outputs.
        """
        directory = self.configuration.get(
            "enfugue.engine.images",
            os.path.join(self.engine_root_dir, "images")
        )
        check_make_directory(directory)
        return directory

    @property
    def engine_intermediate_dir(self) -> str:
        """
        Gets the location for image intermediate outputs.
        """
        directory = self.configuration.get(
            "enfugue.engine.intermediate",
            os.path.join(self.engine_root_dir, "intermediate")
        )
        check_make_directory(directory)
        return directory
    
    @property
    def engine_intermediate_steps(self) -> int:
        """
        Gets the number of steps to wait before decoding an intermediate
        Default to 5; set to 1 to decode every intermediate (not recommended,)
        or set to 0 to disable intermediate.
        """
        return self.configuration.get("enfugue.engine.intermediates", 5)

    @property
    def engine_tensorrt_dir(self) -> str:
        """
        Gets the location for tensorrt engines.
        """
        directory = self.configuration.get("enfugue.engine.tensorrt", os.path.join(self.engine_root_dir, "tensorrt"))
        check_make_directory(directory)
        return directory

    @property
    def engine_checkpoint_dir(self) -> str:
        """
        Returns the engine checkpoint location.
        """
        path = self.configuration.get("enfugue.engine.checkpoint", os.path.join(self.engine_root_dir, "checkpoint"))
        check_make_directory(path)
        return path

    @property
    def default_model_ckpt(self) -> str:
        """
        Returns the location where the default model should be.
        """
        default_model_ckpt = os.path.basename(DEFAULT_MODEL)
        found = find_file_in_directory(self.engine_checkpoint_dir, default_model_ckpt)
        return found if found else os.path.join(self.engine_checkpoint_dir, default_model_ckpt)

    @property
    def default_inpaint_ckpt(self) -> str:
        """
        Returns the location where the default inpaint model should be.
        """
        default_inpaint_ckpt = os.path.basename(DEFAULT_INPAINTING_MODEL)
        found = find_file_in_directory(self.engine_checkpoint_dir, default_inpaint_ckpt)
        return found if found else os.path.join(self.engine_checkpoint_dir, default_inpaint_ckpt)

    @property
    def default_sdxl_ckpt(self) -> str:
        """
        Returns the location where the default sdxl checkpoint should be.
        """
        default_sdxl_ckpt = os.path.basename(DEFAULT_SDXL_MODEL)
        found = find_file_in_directory(self.engine_checkpoint_dir, default_sdxl_ckpt)
        return found if found else os.path.join(self.engine_checkpoint_dir, default_sdxl_ckpt)

    @property
    def default_sdxl_refiner_ckpt(self) -> str:
        """
        Returns the location where the default sdxl_refiner model should be.
        """
        default_sdxl_refiner_ckpt = os.path.basename(DEFAULT_SDXL_REFINER)
        found = find_file_in_directory(self.engine_checkpoint_dir, default_sdxl_refiner_ckpt)
        return found if found else os.path.join(self.engine_checkpoint_dir, default_sdxl_refiner_ckpt)

    @property
    def default_sdxl_inpaint_ckpt(self) -> str:
        """
        Returns the location where the default sdxl_refiner model should be.
        """
        default_sdxl_inpaint_ckpt = os.path.basename(DEFAULT_SDXL_INPAINTING_MODEL)
        found = find_file_in_directory(self.engine_checkpoint_dir, default_sdxl_inpaint_ckpt)
        return found if found else os.path.join(self.engine_checkpoint_dir, default_sdxl_inpaint_ckpt)

    @property
    def pending_default_downloads(self) -> List[Tuple[str, str]]:
        """
        Gets default downloads that need to be started.
        """
        (
            default_model_ckpt,
            default_inpaint_ckpt,
        ) = (
            self.default_model_ckpt,
            self.default_inpaint_ckpt,
        )
        pending = []
        if not os.path.exists(default_model_ckpt) and not self.is_downloading(DEFAULT_MODEL):
            pending.append((DEFAULT_MODEL, default_model_ckpt))
        if not os.path.exists(default_inpaint_ckpt) and not self.is_downloading(DEFAULT_INPAINTING_MODEL):
            pending.append((DEFAULT_INPAINTING_MODEL, default_inpaint_ckpt))
        return pending

    @property
    def pending_xl_downloads(self) -> List[Tuple[str, str]]:
        """
        Gets XL downloads that can be started.
        """
        (
            default_sdxl_ckpt,
            default_sdxl_refiner_ckpt,
            default_sdxl_inpaint_ckpt,
        ) = (
            self.default_sdxl_ckpt,
            self.default_sdxl_refiner_ckpt,
            self.default_sdxl_inpaint_ckpt,
        )
        pending = []
        if not os.path.exists(default_sdxl_ckpt) and not self.is_downloading(DEFAULT_SDXL_MODEL):
            pending.append((DEFAULT_SDXL_MODEL, default_sdxl_ckpt))
        if not os.path.exists(default_sdxl_refiner_ckpt) and not self.is_downloading(DEFAULT_SDXL_REFINER):
            pending.append((DEFAULT_SDXL_REFINER, default_sdxl_refiner_ckpt))
        if not os.path.exists(default_sdxl_inpaint_ckpt) and not self.is_downloading(DEFAULT_SDXL_INPAINTING_MODEL):
            pending.append((DEFAULT_SDXL_INPAINTING_MODEL, default_sdxl_inpaint_ckpt))
        return pending

    @property
    def active_default_downloads(self) -> List[Download]:
        """
        Gets default downloads that are currently underway.
        """
        return [
            download for download in self.active_downloads if download.src in [DEFAULT_MODEL, DEFAULT_INPAINTING_MODEL]
        ]

    @property
    def is_downloading_defaults(self) -> bool:
        """
        Returns true if there is an active download of a default model.
        We queue invocations until these are done, otherwise we might spawn
        a conflicting download process.
        """
        return len(self.active_default_downloads) > 0

    @property
    def max_queued_downloads(self) -> int:
        """
        The maximum number of downloads that can be queued.
        """
        return self.configuration.get("enfugue.downloads.queue", self.DEFAULT_MAX_QUEUED_DOWNLOADS)

    @property
    def max_concurrent_downloads(self) -> int:
        """
        The maximum number of downloads that can go at once.
        """
        return self.configuration.get("enfugue.downloads.concurrent", self.DEFAULT_MAX_CONCURRENT_DOWNLOADS)

    @property
    def max_queued_invocations(self) -> int:
        """
        The maximum number of invocations that can be queued.
        """
        return self.configuration.get("enfugue.queue", self.DEFAULT_MAX_QUEUED_INVOCATIONS)

    @property
    def max_intermediate_age(self) -> int:
        """
        Gets the maximum age of an intermediate image (in seconds)
        After this time, it is eligible for removal.
        """
        return self.configuration.get("enfugue.intermediates.age", 3600)

    @property
    def max_timing_cache_age(self) -> int:
        """
        Gets the maxium age for a TensorRT timing cache, in seconds
        """
        return self.configuration.get("enfugue.tensorrt.age", 60 * 60 * 24 * 30)

    @property
    def active_downloads(self) -> List[Download]:
        """
        Gets a list of active downloads
        """
        return [
            download
            for download_list in self.downloads.values()
            for download in download_list
            if not download.complete and download.started
        ]

    @property
    def remaining_concurrent_downloads(self) -> int:
        """
        Gets the remaining number of download slots.
        """
        return self.max_concurrent_downloads - len(self.active_downloads)

    @property
    def can_start_download(self) -> bool:
        """
        Returns true if there are any download slots remaining.
        """
        return self.remaining_concurrent_downloads > 0

    @property
    def can_queue_download(self) -> bool:
        """
        Retrurns true if there are any download queue slots remaining.
        """
        return len(self.download_queue) < self.max_queued_downloads

    @property
    def can_invoke(self) -> bool:
        """
        Returns true if there is no active invocation.
        """
        return self.active_invocation is None and not self.is_downloading_defaults

    @property
    def can_queue_invocation(self) -> bool:
        """
        Returns true if there are open invocation queue spots.
        """
        return len(self.invocation_queue) < self.max_queued_invocations

    @property
    def engine_status(self) -> str:
        """
        Gets a textual description of the engine status.
        """
        if not self.can_invoke:
            return "busy"
        return "ready" if self.engine.keepalive() else "idle"

    @property
    def status(self) -> SystemManagerStatusDict:
        """
        Gets the status of the manager in a dict.
        """
        return {
            "downloads": {
                "active": len(self.active_downloads),
                "queued": len(self.download_queue),
                "total": sum([len(download_list) for download_list in self.downloads.values()]),
            },
            "invocations": {
                "active": not self.can_invoke,
                "queued": len(self.invocation_queue),
                "total": sum([len(invocation_list) for invocation_list in self.invocations.values()]),
            },
        }

    def is_downloading(self, url: str) -> bool:
        """
        Returns true if there is already an active download process for a URL.
        """
        return len([download for download in self.active_downloads if download.src == url]) > 0

    def download(
        self,
        user_id: int,
        url: str,
        destination: str,
        headers: Dict[str, str] = {},
        parameters: Dict[str, Any] = {},
    ) -> Download:
        """
        Starts a download process.
        """
        if self.is_downloading(url):
            raise BadRequestError(f"Already downloading {url}")

        can_start = self.can_start_download
        can_queue = self.can_queue_download

        if not can_start and not can_queue:
            raise TooManyRequestsError()

        download = Download(url, destination, headers=headers, parameters=parameters, progress=True)
        if user_id not in self.downloads:
            self.downloads[user_id] = []
        self.downloads[user_id].append(download)
        if can_start:
            download.start()
        else:
            self.download_queue.append(download)

        return download

    def cancel_download(self, url: str) -> bool:
        """
        Stops an in-progress download.
        """
        for user_id in self.downloads:
            for download in self.downloads[user_id]:
                if download.src == url and not download.complete:
                    download.cancel()
                    return True
        return False

    def invoke(
        self,
        user_id: int,
        plan: LayeredInvocation,
        ui_state: Optional[str] = None,
        disable_intermediate_decoding: bool = False,
        video_rate: Optional[float] = None,
        video_codec: Optional[str] = None,
        video_format: Optional[str] = None,
        **kwargs: Any,
    ) -> Invocation:
        """
        Starts an invocation.
        """
        can_start = self.can_invoke
        can_queue = self.can_queue_invocation

        if not can_start and not can_queue:
            raise TooManyRequestsError()

        if disable_intermediate_decoding:
            kwargs["decode_nth_intermediate"] = None
        else:
            kwargs["decode_nth_intermediate"] = self.engine_intermediate_steps

        if video_rate is not None:
            kwargs["video_rate"] = video_rate
        if video_codec is not None:
            kwargs["video_codec"] = video_codec
        if video_format is not None:
            kwargs["video_format"] = video_format

        invocation = Invocation(
            engine=self.engine,
            interpolator=self.interpolator,
            plan=plan,
            engine_image_dir=self.engine_image_dir,
            engine_intermediate_dir=self.engine_intermediate_dir,
            ui_state=ui_state,
            **kwargs,
        )

        if can_start:
            invocation.start()
            self.active_invocation = invocation
        else:
            self.invocation_queue.append(invocation)
        if user_id not in self.invocations:
            self.invocations[user_id] = []

        self.invocations[user_id].append(invocation)
        return invocation

    def get_downloads(self, user_id: int) -> List[Download]:
        """
        Gets downloads for a user.
        """
        return self.downloads.get(user_id, [])

    def get_invocations(self, user_id: int) -> List[Invocation]:
        """
        Gets invocations for a user.
        """
        return self.invocations.get(user_id, [])

    def stop_engine(self) -> None:
        """
        stops the engine forcibly.
        """
        if self.active_invocation is not None:
            try:
                self.active_invocation.terminate()
                time.sleep(5)
            except Exception as ex:
                logger.info(f"ignoring exception during invocation termination: {ex}")
            self.active_invocation = None
        self.engine.terminate_process()

    def stop_interpolator(self) -> None:
        """
        stops the interpolator forcibly.
        """
        self.interpolator.terminate_process()

    def clean_intermediates(self) -> None:
        """
        Cleans up intermediate files
        """
        reclaimed_bytes = 0
        for file_name in os.listdir(self.engine_intermediate_dir):
            file_path = os.path.join(self.engine_intermediate_dir, file_name)
            file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            file_age = (datetime.datetime.now() - file_mod_time).total_seconds()
            if file_age > self.max_intermediate_age:
                logger.info(f"Removing intermediate {file_path}")
                reclaimed_bytes += os.path.getsize(file_path)
                os.remove(file_path)
        if reclaimed_bytes > 0:
            logger.info(f"Reclaimed {human_size(reclaimed_bytes)} from intermediates")

    def clean_models(self) -> None:
        """
        Cleans up models, engines, etc.
        """
        reclaimed_bytes = 0
        for model_name in os.listdir(self.engine_tensorrt_dir):
            model_path = os.path.join(self.engine_tensorrt_dir, model_name)
            for stage in ["clip", "unet", "vae", "controlledunet", "controlnet"]:
                model_tensorrt_stage_dir = os.path.join(model_path, stage)
                if os.path.exists(model_tensorrt_stage_dir):
                    for stage_key in os.listdir(model_tensorrt_stage_dir):
                        stage_dir = os.path.join(model_tensorrt_stage_dir, stage_key)
                        if not os.path.isdir(stage_dir):
                            continue
                        stage_files = os.listdir(stage_dir)
                        to_remove = []

                        for file_name in stage_files:
                            file_path = os.path.join(stage_dir, file_name)
                            if file_name == "model.opt.onnx":
                                if "engine.plan" in stage_files:
                                    to_remove.append(file_path)
                            elif file_name == "model.onnx":
                                if "model.opt.onnx" in stage_files:
                                    to_remove.append(file_path)
                            elif file_name == "timing_cache":
                                file_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                                file_age = (datetime.datetime.now() - file_mod_time).total_seconds()
                                if file_age > self.max_timing_cache_age or "engine.plan" in stage_files:
                                    to_remove.append(file_path)
                            elif file_name != "engine.plan" and file_name != "metadata.json":
                                to_remove.append(file_path)

                        for file_path in to_remove:
                            logger.info(f"Removing unneeded engine file {file_path}")
                            reclaimed_bytes += os.path.getsize(file_path)
                            os.remove(file_path)
        if reclaimed_bytes > 0:
            logger.info(f"Reclaimed {human_size(reclaimed_bytes)} from models")

    def do_periodic_tasks(self) -> None:
        """
        Looks at the queues and starts actions if necessary.
        """
        if not self.running:
            raise IOError("Manager stopped, periodic tasks should be ceased.")
        with self.lock:
            available_downloads = self.remaining_concurrent_downloads
            while available_downloads > 0:
                try:
                    next_download = self.download_queue.pop(0)
                    next_download.start()
                    available_downloads -= 1
                except IndexError:
                    break

            self.clean_intermediates()
            self.clean_models()

            if self.active_invocation is not None:
                if self.active_invocation.is_dangling:
                    logger.info("Active invocation appears to be dangling, terminating it.")
                    self.active_invocation.timeout()
                    time.sleep(5)
                elif self.active_invocation.results is None and self.active_invocation.error is None:
                    try:
                        self.active_invocation.poll()
                    except IOError:
                        self.active_invocation = None  # results came in between checking and polling
                else:
                    self.active_invocation = None

            if self.active_invocation is None and not self.is_downloading_defaults:
                try:
                    self.active_invocation = self.invocation_queue.pop(0)
                    self.active_invocation.start()
                except IndexError:
                    for invocation_list in self.invocations.values():
                        for invocation in invocation_list:
                            if invocation.id is None:
                                logger.warning(
                                    "Unstarted invocation found that wasn't on invocation queue. Pushing it back on the queue."
                                )
                                self.invocation_queue.append(invocation)
                            elif invocation.is_dangling:
                                logger.warning("Dangling invocation found, timing it out.")
                                invocation.timeout()
