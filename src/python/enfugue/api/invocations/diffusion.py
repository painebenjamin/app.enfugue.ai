from __future__ import annotations

import re
import traceback

from typing import Any, Optional, List, Dict

from datetime import datetime

from pibble.util.strings import Serializer

from enfugue.api.invocations.base import InvocationMonitor
from enfugue.diffusion.engine import DiffusionEngine
from enfugue.util import logger, get_version

from PIL.PngImagePlugin import PngInfo

__all__ = ["DiffusionInvocationMonitor"]

def get_relative_paths(paths: List[str]) -> List[str]:
    """
    Gets relative paths from a list of paths (os agnostic)
    """
    return ["/".join(re.split(r"/|\\", path)[-2:]) for path in paths]

class DiffusionInvocationMonitor(InvocationMonitor):
    """
    Holds the details for a single invocation
    """
    video_result: Optional[str] = None
    image_results: Optional[List[str]] = None
    last_images: Optional[List[str]] = None
    interpolate_id: Optional[int] = None
    interpolate_result: Any = None

    start_interpolate_time: Optional[datetime] = None
    end_interpolate_time: Optional[datetime] = None

    def __init__(
        self,
        engine: DiffusionEngine,
        communication_timeout: Optional[int] = 180,
        **kwargs: Any,
    ) -> None:
        super(DiffusionInvocationMonitor, self).__init__(
            engine=engine,
            communcation_timeout=communication_timeout,
        )
        self.interpolator = kwargs["interpolator"]
        self.plan = kwargs["plan"]

        self.results_dir = kwargs["engine_image_dir"]
        self.intermediate_dir = kwargs["engine_intermediate_dir"]
        self.ui_state = kwargs.get("ui_state", None)
        self.intermediate_steps = kwargs.get("decode_nth_intermediate", None)
        self.metadata = kwargs.get("metadata", None)
        self.save = kwargs.get("save", True)

        self.video_format = kwargs.get("video_format", "mp4")
        self.video_codec = kwargs.get("video_codec", "avc1")
        self.video_rate = kwargs.get("video_rate", 8.0)

    def start(self) -> None:
        """
        Starts the invocation (locks)
        """
        with self.lock:
            self.start_time = datetime.now()
            payload: Dict[str, Any] = {}
            payload["intermediate_dir"] = self.intermediate_dir
            payload["intermediate_steps"] = self.intermediate_steps
            payload["plan"] = self.plan
            self.id = self.engine.dispatch("plan", payload)

    def _communicate(self) -> None:
        """
        Tries to communicate with the engine to see what's going on.
        """
        super(DiffusionInvocationMonitor, self)._communicate()

        if self.result is not None and self.image_results is None:
            # Complete
            self.image_results = []

            if "images" in self.result:
                is_nsfw = self.result.get("nsfw_content_detected", [])
                for i, image in enumerate(self.result["images"]):
                    if len(is_nsfw) > i and is_nsfw[i]:
                        self.image_results.append("nsfw")
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
                        pnginfo.add_text("EnfugueGenerationData", Serializer.serialize(self.plan.metadata))
                        image.save(image_path, pnginfo=pnginfo)
                        self.image_results.append(image_path)
                    else:
                        self.image_results.append("unsaved")

                if self.plan.animation_frames:
                    if self.plan.interpolate_frames or self.plan.reflect:
                        # Start interpolation
                        self.start_interpolate()
                    else:
                        # Save video
                        try:
                            from enfugue.diffusion.util.video_util import Video
                            video_path = f"{self.results_dir}/{self.uuid}.{self.video_format}"
                            Video(self.result["images"]).save(
                                video_path,
                                rate=self.video_rate,
                                encoder=self.video_codec
                            )
                            self.interpolate_result = video_path
                        except Exception as ex:
                            self.error = ex
                            logger.error(f"Couldn't save video: {ex}")
                            logger.debug(traceback.format_exc())

            if self.metadata is not None and "tensorrt_build" in self.metadata:
                logger.info("TensorRT build complete, terminating engine to start fresh on next invocation.")
                self.engine.terminate_process()

    def start_interpolate(self) -> None:
        """
        Starts the interpolation (is locked when called)
        """
        if self.interpolate_id is not None:
            raise IOError("Interpolation already began.")
        assert isinstance(self.image_results, list), "Must have a list of image results"
        self.interpolate_start_time = datetime.now()
        self.interpolate_id = self.interpolator.dispatch("plan", {
            "reflect": self.plan.reflect,
            "frames": self.plan.interpolate_frames,
            "images": self.image_results,
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
            start_comm = datetime.now()
            last_intermediate = self.interpolator.last_intermediate(self.interpolate_id)
            if last_intermediate is not None:
                for key in ["step", "total", "rate", "task"]:
                    if key in last_intermediate:
                        setattr(self, f"last_{key}", last_intermediate[key])
                self.last_intermediate_time = datetime.now()
            end_comm = (datetime.now() - start_comm).total_seconds()

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
                    self.interpolate_end_time = datetime.now()
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

    def format(self) -> Dict[str, Any]:
        """
        Formats the invocation to a dictionary
        """
        with self.lock:
            if self.id is None or self.start_time is None:
                return {"status": "queued", "uuid": self.uuid}

            if self.error is not None:
                if self.image_results:
                    images = get_relative_paths(self.image_results)
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

            images = None # type: ignore[unreachable]
            video = None
            
            try:
                if self.image_results is not None:
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
                    images = get_relative_paths(self.image_results)
                else:
                    status = "processing"
                    self._communicate()
                    if self.image_results is not None:
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
                        images = get_relative_paths(self.image_results)
                    elif self.last_images is not None:
                        images = get_relative_paths(self.last_images)

            except Exception as ex:
                # Set error and recurse
                self.error = ex
                return self.format()

            if self.end_time is None:
                duration = (datetime.now() - self.start_time).total_seconds()
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
                "type": type(self).__name__,
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
