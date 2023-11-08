from __future__ import annotations

import re
import datetime
import traceback

from typing import Any, Optional, List, Dict
from pibble.util.helpers import resolve
from pibble.util.strings import get_uuid, Serializer

from enfugue.util import logger, get_version
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.diffusion.interpolate import InterpolationEngine
from enfugue.diffusion.invocation import LayeredInvocation
from multiprocessing import Lock

from PIL.PngImagePlugin import PngInfo

__all__ = ["Invocation"]


class TerminatedError(Exception):
    pass

def get_relative_paths(paths: List[str]) -> List[str]:
    """
    Gets relative paths from a list of paths (os agnostic)
    """
    return ["/".join(re.split(r"/|\\", path)[-2:]) for path in paths]

class Invocation:
    """
    Holds the details for a single invocation
    """
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    last_intermediate_time: Optional[datetime.datetime]
    results: Optional[List[str]]
    video_result: Optional[str]
    last_images: Optional[List[str]]
    last_step: Optional[int]
    last_total: Optional[int]
    last_rate: Optional[float]
    id: Optional[int]
    error: Optional[Exception]

    def __init__(
        self,
        engine: DiffusionEngine,
        interpolator: InterpolationEngine,
        plan: LayeredInvocation,
        engine_image_dir: str,
        engine_intermediate_dir: str,
        ui_state: Optional[str] = None,
        decode_nth_intermediate: Optional[int] = 10,
        communication_timeout: Optional[int] = 180,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = True,
        video_format: str = "mp4",
        video_codec: str = "avc1",
        video_rate: float = 8.0,
        **kwargs: Any,
    ) -> None:
        self.lock = Lock()
        self.uuid = get_uuid()
        self.engine = engine
        self.interpolator = interpolator
        self.plan = plan

        self.results_dir = engine_image_dir
        self.intermediate_dir = engine_intermediate_dir
        self.ui_state = ui_state
        self.intermediate_steps = decode_nth_intermediate
        self.communication_timeout = communication_timeout
        self.metadata = metadata
        self.save = save

        self.video_format = video_format
        self.video_codec = video_codec
        self.video_rate = video_rate

        self.id = None
        self.interpolate_id = None
        self.error = None

        self.start_time = None
        self.end_time = None
        self.start_interpolate_time = None
        self.end_interpolate_time = None
        self.last_intermediate_time = None

        self.last_step = None
        self.last_total = None
        self.last_rate = None
        self.last_images = None
        self.last_task = None
        self.results = None
        self.interpolate_result = None

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

                    if self.plan.animation_frames:
                        if self.plan.interpolate_frames or self.plan.reflect:
                            # Start interpolation
                            self.start_interpolate()
                        else:
                            # Save video
                            try:
                                from enfugue.diffusion.util.video_util import Video
                                video_path = f"{self.results_dir}/{self.uuid}.{self.video_format}"
                                Video(result["images"]).save(
                                    video_path,
                                    rate=self.video_rate,
                                    encoder=self.video_codec
                                )
                                self.interpolate_result = video_path # type: ignore[assignment]
                            except Exception as ex:
                                self.error = ex
                                logger.error(f"Couldn't save video: {ex}")
                                logger.debug(traceback.format_exc())

                if "error" in result:
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
            payload = self.plan.serialize(self.intermediate_dir)
            payload["intermediate_dir"] = self.intermediate_dir
            payload["intermediate_steps"] = self.intermediate_steps
            self.id = self.engine.dispatch("plan", payload)

    def poll(self) -> None:
        """
        Calls communicate once (locks)
        """
        with self.lock:
            self._communicate()

    def start_interpolate(self) -> None:
        """
        Starts the interpolation (is locked when called)
        """
        from PIL import Image
        if self.interpolate_id is not None:
            raise IOError("Interpolation already began.")
        assert isinstance(self.results, list), "Must have a list of image results"
        self.interpolate_start_time = datetime.datetime.now()
        self.interpolate_id = self.interpolator.dispatch("plan", {
            "reflect": self.plan.reflect,
            "frames": self.plan.interpolate_frames,
            "images": [
                Image.open(path) for path in self.results
            ],
            "save_path": f"{self.results_dir}/{self.uuid}.{self.video_format}",
            "video_rate": self.video_rate,
            "video_codec": self.video_codec
        })

    def _interpolate_communicate(self) -> None:
        """
        Tries to communicate with the engine to see what's going on.
        """
        if self.interpolate_id is None:
            raise IOError("Interpolation not started yet.")
        if self.interpolate_result is not None: # type: ignore[unreachable]
            raise IOError("Interpolation already completed.")
        try:
            start_comm = datetime.datetime.now()
            last_intermediate = self.interpolator.last_intermediate(self.interpolate_id)
            if last_intermediate is not None:
                for key in ["step", "total", "rate", "task"]:
                    if key in last_intermediate:
                        setattr(self, f"last_{key}", last_intermediate[key])
                self.last_intermediate_time = datetime.datetime.now()
            end_comm = (datetime.datetime.now() - start_comm).total_seconds()

            try:
                result = self.interpolator.wait(self.interpolate_id, timeout=0.1)
            except TimeoutError:
                raise
            except Exception as ex:
                result = None
                self.error = ex

            if result is not None:
                # Complete
                if isinstance(result, list):
                    from enfugue.diffusion.util.video_util import Video
                    self.interpolate_end_time = datetime.datetime.now()
                    video_path = f"{self.results_dir}/{self.uuid}.{self.video_format}"
                    Video(result).save(
                        video_path,
                        rate=self.video_rate,
                        encoder=self.video_codec
                    )
                    self.interpolate_result = video_path
                else:
                    self.interpolate_result = result
        except TimeoutError:
            return

    def poll_interpolator(self) -> None:
        """
        Calls communicate on the interpolator once (locks)
        """
        with self.lock:
            self._interpolate_communicate()

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
                if self.results:
                    images = get_relative_paths(self.results)
                else:
                    images = None
                if self.interpolate_result:
                    video = get_relative_paths([self.interpolate_result])[0] # type: ignore[unreachable]
                else:
                    video = None
                if self.end_time is not None:
                    duration = (self.end_time - self.start_time).total_seconds()
                else:
                    duration = 0
                return {
                    "status": "error",
                    "uuid": self.uuid,
                    "message": str(self.error),
                    "images": images,
                    "video": video,
                    "duration": duration,
                }

            images = None
            video = None

            if self.results is not None:
                if self.plan.animation_frames:
                    if self.interpolate_result:
                        status = "completed" # type: ignore[unreachable]
                        video = get_relative_paths([self.interpolate_result])[0]
                    elif self.plan.interpolate_frames or self.plan.reflect:
                        status = "interpolating"
                        self._interpolate_communicate()
                    else:
                        # Saving
                        status = "processing"
                else:
                    status = "completed"
                    if self.plan.animation_frames:
                        video = get_relative_paths([self.interpolate_result])[0] # type: ignore[list-item]
                images = get_relative_paths(self.results)
            else:
                status = "processing"
                self._communicate()
                if self.results is not None:
                    # Finished in previous _communicate() calling
                    if self.plan.animation_frames: # type: ignore[unreachable]
                        if self.plan.interpolate_frames or self.plan.reflect:
                            # Interpolation just started
                            ...
                        elif self.interpolate_result:
                            status = "completed"
                            video = get_relative_paths([self.interpolate_result])[0]
                    else:
                        status = "completed"
                    images = get_relative_paths(self.results)
                elif self.last_images is not None:
                    images = get_relative_paths(self.last_images)

            if self.end_time is None:
                duration = (datetime.datetime.now() - self.start_time).total_seconds()
            else:
                duration = (self.end_time - self.start_time).total_seconds()

            step, total, progress, rate = None, None, None, None
            if self.last_total is not None and self.last_total > 0:
                total = self.last_total
            if self.last_step is not None:
                step = self.last_total if status == "completed" else self.last_step
            if total is not None and step is not None:
                progress = step / total
            if self.last_rate is not None:
                rate = self.last_rate
            elif step is not None:
                rate = step / duration

            if video:
                video = f"animation/{video}"

            formatted = {
                "id": self.id,
                "uuid": self.uuid,
                "status": status,
                "progress": progress,
                "step": step,
                "duration": duration,
                "total": total,
                "images": images,
                "video": video,
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
