from __future__ import annotations

import os
import numpy as np

from typing import (
    Any,
    Iterator,
    Type,
    Union,
    List,
    Tuple,
    Iterable,
    Dict,
    Optional,
    Callable,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from enfugue.diffusion.interpolate.model import InterpolatorModel # type: ignore[attr-defined]

from PIL import Image
from datetime import datetime

from enfugue.diffusion.engine import Engine
from enfugue.diffusion.process import EngineProcess
from enfugue.util import (
    get_frames_or_image_from_file,
    get_frames_or_image,
    check_make_directory,
    logger
)

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

__all__ = ["InterpolationEngine"]

class InterpolatorEngineProcess(EngineProcess):
    """
    Capture the interpolator in a process because tensorflow has a lot of global state.
    """
    interpolator_path: str

    @property
    def interpolator(self) -> InterpolatorModel:
        """
        Gets or instantiates the interpolator
        """
        if not hasattr(self, "_interpolator"):
            if getattr(self, "interpolator_path", None) is None:
                raise IOError("Can't get interpolator - path was never sent to process.")
            logger.debug(f"Loading interpolator from {self.interpolator_path}")
            from enfugue.diffusion.interpolate.model import InterpolatorModel # type: ignore[attr-defined]
            self._interpolator = InterpolatorModel(self.interpolator_path)
        return self._interpolator

    def interpolate_recursive(
        self,
        frames: Iterable[Image.Image],
        multiplier: Union[int, Tuple[int, ...]] = 2,
    ) -> Iterator[Image]:
        """
        Provides a generator for interpolating between multiple frames.
        """
        if isinstance(multiplier, tuple) or isinstance(multiplier, list): # type: ignore[unreachable]
            if len(multiplier) == 1:
                multiplier = multiplier[0]
            else:
                this_multiplier = multiplier[0]
                recursed_multiplier = multiplier[1:]
                for frame in self.interpolate_recursive(
                    frames=self.interpolate_recursive(
                        frames=frames, # type: ignore[arg-type]
                        multiplier=recursed_multiplier,
                    ),
                    multiplier=this_multiplier,
                ):
                    yield frame
                return

        previous_frame = None
        frame_index = 0
        for frame in frames:
            frame_index += 1
            if previous_frame is not None:
                for i in range(multiplier - 1): # type: ignore[unreachable]
                    yield self.interpolate(
                        previous_frame,
                        frame,
                        (i + 1) / multiplier
                    )
            yield frame
            previous_frame = frame
            frame_start = datetime.now()

    def loop(
        self,
        frames: Iterable[Image.Image],
        ease_frames: int = 2,
        double_ease_frames: int = 1,
        hold_frames: int = 0,
        trigger_callback: Optional[Callable[[Image.Image], Image]] = None,
    ) -> Iterable[Image]:
        """
        Takes a video and creates a gently-looping version of it.
        """
        if trigger_callback is None:
            trigger_callback = lambda image: image

        # Memoized frames
        frame_list: List[Image.Image] = [frame for frame in frames]
        
        if double_ease_frames:
            double_ease_start_frames, frame_list = frame_list[:double_ease_frames], frame_list[double_ease_frames:]
        else:
            double_ease_start_frames = []
        if ease_frames:
            ease_start_frames, frame_list = frame_list[:ease_frames], frame_list[ease_frames:]
        else:
            ease_start_frames = []

        if double_ease_frames:
            frame_list, double_ease_end_frames = frame_list[:-double_ease_frames], frame_list[-double_ease_frames:]
        else:
            double_ease_end_frames = []
        if ease_frames:
            frame_list, ease_end_frames = frame_list[:-ease_frames], frame_list[-ease_frames:]
        else:
            ease_end_frames = []

        # Interpolate frames
        double_ease_start_frames = [
            trigger_callback(frame) for frame in self.interpolate_recursive(
                frames=double_ease_start_frames,
                multiplier=(2,2),
            )
        ]
        ease_start_frames = [
            trigger_callback(frame) for frame in self.interpolate_recursive(
                frames=ease_start_frames,
                multiplier=2,
            )
        ]
        ease_end_frames = [
            trigger_callback(frame) for frame in self.interpolate_recursive(
                frames=ease_end_frames,
                multiplier=2,
            )
        ]
        double_ease_end_frames = [
            trigger_callback(frame) for frame in self.interpolate_recursive(
                frames=double_ease_end_frames,
                multiplier=(2,2),
            )
        ]

        # Return to one list
        frame_list = double_ease_start_frames + ease_start_frames + frame_list + ease_end_frames + double_ease_end_frames

        # Iterate
        for frame in frame_list:
            yield frame

        # Hold on final frame
        for i in range(hold_frames):
            yield frame_list[-1]

        # Reverse the frames
        frame_list.reverse()
        for frame in frame_list[1:-1]:
            yield frame

        # Hold on first frame
        for i in range(hold_frames):
            yield frame_list[-1]

    def handle_plan(
        self,
        instruction_id: int,
        instruction_payload: Dict[str, Any]
    ) -> Union[str, List[Image]]:
        """
        Handles an entire video potentially with recursion
        """
        interpolate_frames = instruction_payload["frames"]
        if isinstance(interpolate_frames, list):
            interpolate_frames = tuple(interpolate_frames)

        images = instruction_payload["images"]
        if isinstance(images, str):
            images = get_frames_or_image_from_file(images)
        elif isinstance(images, Image.Image):
            images = get_frames_or_image(images)

        image_count = len(images)
        interpolated_count = image_count
        if isinstance(interpolate_frames, tuple):
            for multiplier in interpolate_frames:
                interpolated_count *= multiplier
        else:
            interpolated_count *= interpolate_frames

        reflect = instruction_payload.get("reflect", False)
        ease_frames = instruction_payload.get("ease_frames", 2)
        double_ease_frames = instruction_payload.get("double_ease_frames", 1)
        hold_frames = instruction_payload.get("hold_frames", 0)

        frame_complete = 0
        frame_start = datetime.now()
        frame_times: List[float] = []

        if reflect:
            interpolated_count += ease_frames + (2 * double_ease_frames)

        def trigger_callback(image: Image.Image) -> Image.Image:
            """
            Triggers the callback, which sends progress back up the line
            """
            nonlocal frame_complete
            nonlocal frame_start
            nonlocal frame_times
            frame_time = datetime.now()
            frame_seconds = (frame_time - frame_start).total_seconds()
            frame_times.append(frame_seconds)
            frame_complete += 1
            frame_rate = (sum(frame_times[-8:]) / min(frame_complete, 8))

            if frame_complete % 8 == 0:
                logger.debug(f"Completed {frame_complete}/{interpolated_count} frames (average {frame_rate} sec/frame)")

            self.intermediates.put_nowait({
                "id": instruction_id,
                "step": frame_complete,
                "total": interpolated_count,
                "rate": None if not frame_rate else 1.0 / frame_rate,
                "task": "Interpolating"
            })
            frame_start = frame_time
            return image

        if interpolate_frames:
            logger.debug(f"Beginning interpolation - will interpolate {image_count} frames with interpolation amount(s) [{interpolate_frames}] (a total of {interpolated_count} frames")
            images = [
                trigger_callback(img) for img in
                self.interpolate_recursive(
                    frames=images,
                    multiplier=interpolate_frames,
                )
            ]
        elif reflect:
            interpolated_count -= image_count # Small interpolation amount

        if reflect:
            logger.debug(f"Beginning reflection, will interpolate {double_ease_frames} frame(s) twice, {ease_frames} frame(s) once and hold {hold_frames} frame(s).")
            images = self.loop(
                frames=images,
                ease_frames=ease_frames,
                double_ease_frames=double_ease_frames,
                hold_frames=hold_frames,
                trigger_callback=trigger_callback
            )

        if "save_path" in instruction_payload:
            from enfugue.diffusion.util.video_util import Video
            Video(images).save(
                instruction_payload["save_path"],
                rate=instruction_payload.get("video_rate", 8.0),
                encoder=instruction_payload.get("video_codec", "avc1"),
                overwrite=True
            )
            return instruction_payload["save_path"]
        return images

    def interpolate(
        self,
        left: Image,
        right: Image,
        alpha: float
    ) -> Image:
        """
        Executes an individual interpolation.
        """
        left_data = np.asarray(left.convert("RGB")).astype(np.float32) / _UINT8_MAX_F
        right_data = np.asarray(right.convert("RGB")).astype(np.float32) / _UINT8_MAX_F
        mid_data = self.interpolator(
            np.expand_dims(left_data, axis=0),
            np.expand_dims(right_data, axis=0),
            np.full(shape=(1,), fill_value=alpha, dtype=np.float32)
        )[0]
        mid_data = (
            np.clip(mid_data * _UINT8_MAX_F, 0.0, _UINT8_MAX_F) + 0.5
        ).astype(np.uint8)
        return Image.fromarray(mid_data)

    def handle(
        self,
        instruction_id: int,
        instruction_action: str,
        instruction_payload: Any
    ) -> Any:
        """
        Processes two images and returns the interpolated image.
        """
        if not isinstance(instruction_payload, dict):
            raise IOError("Expected dictionary payload")
        
        if "path" in instruction_payload:
            self.interpolator_path = instruction_payload["path"]

        if instruction_action == "plan":
            return self.handle_plan(
                instruction_id=instruction_id,
                instruction_payload=instruction_payload
            )

        to_process = []

        if instruction_action == "process":
            left = instruction_payload["left"]
            right = instruction_payload["right"]
            alpha = instruction_payload["alpha"]
            to_process.append((left, right, alpha))
        elif instruction_action == "batch":
            for image_dict in instruction_payload["batch"]:
                left = image_dict["left"]
                right = image_dict["right"]
                alpha = image_dict["alpha"]
                to_process.append((left, right, alpha))

        results = []
        for left, right, alpha in to_process:
            if isinstance(alpha, list):
                results.append([
                    self.interpolate(left, right, a)
                    for a in alpha
                ])
            else:
                results.append(self.interpolate(left, right, alpha))

        return results

class InterpolationEngine(Engine):
    """
    Manages the interpolate in a sub-process
    """
    
    STYLE_MODEL_REPO = "akhaliq/frame-interpolation-film-style"

    @property
    def process_class(self) -> Type[EngineProcess]:
        """
        Override to pass interpolator process
        """
        return InterpolatorEngineProcess

    @property
    def model_dir(self) -> str:
        """
        Gets the model directory from config
        """
        path = self.configuration.get("enfugue.engine.other", "~/.cache/enfugue/other")
        if path.startswith("~"):
            path = os.path.expanduser(path)
        path = os.path.realpath(path)
        check_make_directory(path)
        return path

    @property
    def style_model_path(self) -> str:
        """
        Gets the style model path
        """
        if not hasattr(self, "_style_model_path"):
            from huggingface_hub import snapshot_download
            self._style_model_path = os.path.join(
                self.model_dir,
                "models--" + self.STYLE_MODEL_REPO.replace("/", "--")
            )
            if not os.path.exists(self._style_model_path):
                os.makedirs(self._style_model_path)
            if not os.path.exists(os.path.join(self._style_model_path, "saved_model.pb")):
                snapshot_download(
                    self.STYLE_MODEL_REPO,
                    local_dir=self._style_model_path,
                    local_dir_use_symlinks=False
                )
        return self._style_model_path

    def dispatch(
        self,
        action: str,
        payload: Any = None,
        spawn_process: bool = True
    ) -> Any:
        """
        Intercept dispatch to inject path
        """
        if isinstance(payload, dict) and "path" not in payload:
            payload["path"] = self.style_model_path
        return super(InterpolationEngine, self).dispatch(
            action,
            payload,
            spawn_process
        )

    def __call__(
        self,
        images: List[Image],
        interpolate_frames: Union[int, Tuple[int]],
    ) -> List[Image]:
        """
        Executes interpolation.
        """
        return self.invoke(
            "recursive",
            {
                "path": self.style_model_path,
                "images": images,
                "frames": interpolate_frames
            }
        )
