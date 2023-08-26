import re
import datetime

from typing import Any, Optional, List, Dict
from pibble.util.helpers import resolve
from pibble.util.strings import get_uuid, Serializer

from enfugue.util import logger, get_version
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.diffusion.plan import DiffusionPlan
from multiprocessing import Lock

from PIL.PngImagePlugin import PngInfo

__all__ = ["Invocation"]


class TerminatedError(Exception):
    pass


class Invocation:
    """
    Holds the details for a single invocation
    """

    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    last_intermediate_time: Optional[datetime.datetime]
    results: Optional[List[str]]
    last_images: Optional[List[str]]
    last_step: Optional[int]
    last_total: Optional[int]
    last_rate: Optional[float]
    id: Optional[int]
    error: Optional[Exception]

    def __init__(
        self,
        engine: DiffusionEngine,
        plan: DiffusionPlan,
        engine_image_dir: str,
        engine_intermediate_dir: str,
        ui_state: Optional[str] = None,
        decode_nth_intermediate: Optional[int] = 10,
        communication_timeout: Optional[int] = 180,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
        **kwargs: Any,
    ) -> None:
        self.lock = Lock()
        self.uuid = get_uuid()
        self.engine = engine
        self.plan = plan

        self.results_dir = engine_image_dir
        self.intermediate_dir = engine_intermediate_dir
        self.ui_state = ui_state
        self.intermediate_steps = decode_nth_intermediate
        self.communication_timeout = communication_timeout
        self.metadata = metadata
        self.save = save

        self.id = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.last_intermediate_time = None
        self.last_step = None
        self.last_total = None
        self.last_rate = None
        self.last_images = None
        self.last_task = None
        self.results = None

    def _communicate(self) -> None:
        """
        Tries to communicate with the engine to see what's going on.
        """
        if self.id is None:
            raise IOError("Invocation not started yet.")
        if self.results is not None:
            raise IOError("Invocation already completed.")
        try:
            start_comm = datetime.datetime.now()
            last_intermediate = self.engine.last_intermediate(self.id)
            if last_intermediate is not None:
                for key in ["step", "total", "images", "rate", "task"]:
                    if key in last_intermediate:
                        setattr(self, f"last_{key}", last_intermediate[key])
                self.last_intermediate_time = datetime.datetime.now()
            end_comm = (datetime.datetime.now() - start_comm).total_seconds()
            try:
                result = self.engine.wait(self.id, timeout=0.1)
            except TimeoutError:
                raise
            except Exception as ex:
                result = None
                self.error = ex
            if result is not None:
                # Complete
                self.results = []
                self.end_time = datetime.datetime.now()
                if "images" in result:
                    is_nsfw = result.get("nsfw_content_detected", [])
                    for i, image in enumerate(result["images"]):
                        if len(is_nsfw) > i and is_nsfw[i]:
                            self.results.append("nsfw")
                        elif self.save:
                            image_path = f"{self.results_dir}/{self.uuid}_{i}.png"
                            pnginfo = PngInfo()
                            image_text_metadata = getattr(image, "text", {})
                            for key in image_text_metadata:
                                if key not in ["EnfugueVersion", "EnfugueUIState"]:
                                    pnginfo.add_text(key, image_text_metadata[key])
                            pnginfo.add_text("EnfugueVersion", f"{get_version()}")
                            if self.ui_state is not None:
                                pnginfo.add_text("EnfugueUIState", Serializer.serialize(self.ui_state))
                            image.save(image_path, pnginfo=pnginfo)
                            self.results.append(image_path)
                        else:
                            self.results.append("unsaved")
                elif "error" in result:
                    error_type = resolve(result["error"])
                    self.error = error_type(result["message"])
                    if "traceback" in result:
                        logger.error(f"Traceback for invocation {self.uuid}:")
                        logger.debug(result["traceback"])
                if self.metadata is not None and "tensorrt_build" in self.metadata:
                    logger.info("TensorRT build complete, terminating engine to start fresh on next invocation.")
                    self.engine.terminate_process()

        except TimeoutError:
            return

    def _check_raise_error(self) -> None:
        """
        Raises an error if one has been set.
        """
        if self.error is not None:
            raise self.error

    def start(self) -> None:
        """
        Starts the invocation (locks)
        """
        with self.lock:
            self.start_time = datetime.datetime.now()
            payload = self.plan.get_serialization_dict(self.intermediate_dir)
            payload["intermediate_dir"] = self.intermediate_dir
            payload["intermediate_steps"] = self.intermediate_steps
            self.id = self.engine.dispatch("plan", payload)

    def poll(self) -> None:
        """
        Calls communicate once (locks)
        """
        with self.lock:
            self._communicate()

    @property
    def is_dangling(self) -> bool:
        """
        Determine if this invocation appears lost.
        """
        if self.id is None or self.start_time is None:
            return False
        if self.end_time is not None and self.results is not None or self.error is not None:
            return False
        if self.communication_timeout is None:
            return False
        if self.last_intermediate_time is not None:
            last_known_time = self.last_intermediate_time
        else:
            last_known_time = self.start_time
        seconds_since_last_communication = (datetime.datetime.now() - last_known_time).total_seconds()
        return seconds_since_last_communication > self.communication_timeout

    def timeout(self) -> None:
        """
        Times out an invocation that got lost
        """
        self.error = TimeoutError("Invocation timed out.")
        self.end_time = datetime.datetime.now()

    def terminate(self) -> None:
        """
        Kills an active invocation
        """
        if self.id is None:
            raise IOError("Invocation not started yet.")
        if self.results is not None:
            raise IOError("Invocation completed.")
        if self.error is None:
            self.error = TerminatedError("The invocation was terminated prematurely")
        else:
            raise IOError(f"Invocation already ended in error {self.error}")

    def format(self) -> Dict[str, Any]:
        """
        Formats the invocation to a dictionary
        """
        with self.lock:
            if self.id is None or self.start_time is None:
                return {"status": "queued", "uuid": self.uuid}

            if self.error is not None:
                return {
                    "status": "error",
                    "uuid": self.uuid,
                    "message": str(self.error),
                }

            images = None
            if self.results is not None:
                status = "completed"
                images = ["/".join(re.split(r"/|\\", path)[-2:]) for path in self.results]
            else:
                status = "processing"
                self._communicate()
                if self.last_images is not None:
                    images = ["/".join(re.split(r"/|\\", path)[-2:]) for path in self.last_images]

            if self.end_time is None:
                duration = (datetime.datetime.now() - self.start_time).total_seconds()
            else:
                duration = (self.end_time - self.start_time).total_seconds()

            step, total, progress, rate = None, None, None, None
            if self.last_total is not None and self.last_total > 0:
                total = self.last_total
            if self.last_step is not None:
                step = total if self.results is not None else self.last_step
            if total is not None and step is not None:
                progress = step / total
            if self.last_rate is not None:
                rate = self.last_rate
            elif step is not None:
                rate = step / duration

            formatted = {
                "id": self.id,
                "uuid": self.uuid,
                "status": status,
                "progress": progress,
                "step": step,
                "duration": duration,
                "total": total,
                "images": images,
                "rate": rate,
                "task": self.last_task
            }
            if self.metadata:
                formatted["metadata"] = self.metadata
            return formatted

    def __str__(self) -> str:
        """
        Stringifies the invocation for debugging.
        """

        return f"Invocation {self.uuid}, last step: {self.last_step}, last total: {self.last_total}: error: {self.error}, results: {self.results}"
