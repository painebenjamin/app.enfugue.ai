from __future__ import annotations

import os
import logging
import datetime
import traceback

import PIL
import PIL.Image

from typing import (
    Any,
    Optional,
    Callable,
    Union,
    List,
    Dict,
    Tuple,
    TYPE_CHECKING,
)

from multiprocessing import Process
from multiprocessing.queues import Queue
from queue import Empty

from pibble.api.configuration import APIConfiguration
from pibble.util.helpers import qualify
from pibble.util.strings import Serializer, get_uuid

from enfugue.util import logger

if TYPE_CHECKING:
    # We only import these here when type checking.
    # We avoid importing them before the process starts at runtime,
    # since we don't want torch to initialize itself.
    from enfugue.diffusion.manager import DiffusionPipelineManager
    from enfugue.diffusion.plan import DiffusionPlan

__all__ = ["DiffusionEngineProcess"]


class DiffusionEngineProcess(Process):
    """
    This process allows for easy two-way communication with a waiting
    Stable Diffusion Pipeline. Torch is only initiated after the process
    has began.
    """

    POLLING_DELAY_MS = 500
    IDLE_SEC = 15

    def __init__(
        self,
        instructions: Queue,
        results: Queue,
        intermediates: Queue,
        configuration: Optional[APIConfiguration] = None,
    ) -> None:
        super(DiffusionEngineProcess, self).__init__()
        self.configuration = APIConfiguration()
        self.instructions = instructions
        self.results = results
        self.intermediates = intermediates
        if configuration is not None:
            self.configuration = configuration

    @property
    def pipemanager(self) -> DiffusionPipelineManager:
        """
        Gets the model manager.
        """
        if not hasattr(self, "_pipemanager"):
            from enfugue.diffusion.manager import DiffusionPipelineManager

            self._pipemanager = DiffusionPipelineManager(self.configuration)
        return self._pipemanager

    @property
    def idle_seconds(self) -> int:
        """
        Gets the maximum number of seconds to go idle before exiting.
        """
        return self.configuration.get("enfugue.idle", self.IDLE_SEC)

    def get_diffusion_plan(self, payload: Dict[str, Any]) -> DiffusionPlan:
        """
        Deserializes a plan.
        """
        from enfugue.diffusion.plan import DiffusionPlan

        return DiffusionPlan.deserialize_dict(payload)

    def execute_diffusion_plan(
        self,
        instruction_id: int,
        plan: DiffusionPlan,
        intermediate_dir: Optional[str] = None,
        intermediate_steps: Optional[int] = None,
    ) -> List[PIL.Image.Image]:
        """
        Executes the plan, getting callbacks first.
        """

        progress_callback = self.create_progress_callback(instruction_id)
        task_callback = self.create_task_callback(instruction_id)
        if intermediate_dir is not None:
            image_callback = self.create_image_callback(instruction_id, intermediate_dir=intermediate_dir)
        else:
            image_callback = None

        self.pipemanager.keepalive_callback = lambda: progress_callback(0, 0, 0.0)

        return plan.execute(
            self.pipemanager,
            image_callback=image_callback,
            image_callback_steps=intermediate_steps,
            progress_callback=progress_callback,
            task_callback=task_callback
        )

    def create_progress_callback(
        self,
        instruction_id: int,
    ) -> Callable[[int, int, float], None]:
        """
        Generates a callback that sends progress to the pipe.
        """

        def callback(current_step: int, total_steps: int, current_rate: float) -> None:
            to_send: Dict[str, Any] = {
                "id": instruction_id,
                "step": current_step,
                "total": total_steps,
                "rate": current_rate,
            }
            self.intermediates.put_nowait(Serializer.serialize(to_send))

        return callback

    def create_image_callback(
        self, instruction_id: int, intermediate_dir: str
    ) -> Callable[[List[PIL.Image.Image]], None]:
        """
        Generates a callback that sends decoded latents to the pipe, if asked.
        """

        def callback(images: List[PIL.Image.Image]) -> None:
            image_id = get_uuid()
            to_send: Dict[str, Any] = {"id": instruction_id, "images": []}
            for i, image in enumerate(images):
                image_path = os.path.join(intermediate_dir, f"{instruction_id}_{image_id}_{i}.png")
                image.save(image_path)
                to_send["images"].append(image_path)
            self.intermediates.put_nowait(Serializer.serialize(to_send))

        return callback

    def create_task_callback(
        self,
        instruction_id: int
    ) -> Callable[[str], None]:
        """
        Creates a callback that sends the current task to the pipe (for multi-step plans.)
        """

        def callback(task: str) -> None:
            logger.debug(f"Instruction {instruction_id} beginning task “{task}”")
            payload = {"id": instruction_id, "task": task}
            self.intermediates.put_nowait(Serializer.serialize(payload))
        
        return callback

    def check_invoke_kwargs(
        self,
        instruction_id: int,
        model: Optional[str] = None,
        refiner: Optional[str] = None,
        inpainter: Optional[str] = None,
        lora: Optional[Union[str, Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        controlnet: Optional[str] = None,
        vae: Optional[str] = None,
        control_image: Optional[Union[str, PIL.Image.Image]] = None,
        seed: Optional[int] = None,
        image_callback_steps: Optional[int] = None,
        build_tensorrt: Optional[bool] = None,
        intermediate_dir: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        chunking_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        size: Optional[int] = None,
        refiner_size: Optional[int] = None,
        inpainter_size: Optional[int] = None,
        process_control_image: bool = True,
        **kwargs: Any,
    ) -> dict:
        """
        Sets local vars which will rebuild the pipeline if required
        """
        kwargs["progress_callback"] = self.create_progress_callback(instruction_id)
        kwargs["task_callback"] = self.create_task_callback(instruction_id)
        if intermediate_dir is not None and image_callback_steps is not None:
            kwargs["latent_callback"] = self.create_image_callback(instruction_id, intermediate_dir=intermediate_dir)
            kwargs["latent_callback_steps"] = image_callback_steps
            kwargs["latent_callback_type"] = "pil"
        else:
            kwargs["latent_callback"] = None

        self.pipemanager.keepalive_callback = lambda: kwargs["progress_callback"](0, 0, 0.0)

        if model is not None:
            self.pipemanager.model = model  # type: ignore

        if refiner is not None:
            self.pipemanager.refiner = refiner  # type: ignore

        if inpainter is not None:
            self.pipemanager.inpainter = inpainter  # type: ignore

        if vae is not None:
            self.pipemanager.vae = vae  # type: ignore

        if seed is not None:
            self.pipemanager.seed = seed

        if lora is not None:
            self.pipemanager.lora = lora  # type: ignore

        if inversion is not None:
            self.pipemanager.inversion = inversion  # type: ignore

        if build_tensorrt is not None:
            self.pipemanager.build_tensorrt = build_tensorrt

        if controlnet is not None:
            self.pipemanager.controlnet = controlnet  # type: ignore
            if control_image is not None:
                if isinstance(control_image, str):
                    control_image = PIL.Image.open(control_image)
                if process_control_image:
                    if controlnet == "canny":
                        control_image = self.pipemanager.edge_detector.canny(control_image)
                    elif controlnet == "hed":
                        control_image = self.pipemanager.edge_detector.hed(control_image)
                    elif controlnet == "scribble":
                        control_image = self.pipemanager.edge_detector.hed(control_image, scribble=True)
                    elif controlnet == "pidi":
                        control_image = self.pipemanager.edge_detector.pidi(control_image)
                    elif controlnet == "depth":
                        control_image = self.pipemanager.depth_detector.midas(control_image)
                    elif controlnet == "normal":
                        control_image = self.pipemanager.depth_detector.normal(control_image)
                    elif controlnet == "pose":
                        control_image = self.pipemanager.pose_detector.detect(control_image)
                    elif controlnet == "line":
                        control_image = self.pipemanager.line_detector.detect(control_image)
                    elif controlnet == "anime":
                        control_image = self.pipemanager.line_detector.detect(control_image, anime=True)
                    elif controlnet == "mlsd":
                        control_image = self.pipemanager.line_detector.mlsd(control_image)
                kwargs["control_image"] = control_image

        if size is not None:
            self.pipemanager.size = size

        if refiner_size is not None:
            self.pipemanager.refiner_size = refiner_size

        if inpainter_size is not None:
            self.pipemanager.inpainter_size = inpainter_size

        if width is not None:
            kwargs["width"] = int(width)

        if height is not None:
            kwargs["height"] = int(height)

        if chunking_size is not None:
            kwargs["chunking_size"] = int(chunking_size)

        if guidance_scale is not None:
            kwargs["guidance_scale"] = float(guidance_scale)

        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = int(num_inference_steps)

        return kwargs

    def clear_intermediates(self, instruction_id: int) -> None:
        """
        Clears intermediates for a specific instruction ID.
        """
        try:
            while True:
                next_intermediate = self.intermediates.get_nowait()
                # Avoid parsing
                if f'"id": {instruction_id}' not in next_intermediate[:40]:
                    # Not ours, put back on the queue
                    self.intermediates.put_nowait(next_intermediate)
        except Empty:
            return

    def clear_responses(self, instruction_id: int) -> None:
        """
        Clears responses for a specific instruction ID
        """
        try:
            while True:
                next_result = self.results.get_nowait()
                # Avoid parsing
                if f'"id": {instruction_id}' not in next_result[:40]:
                    # Not ours, put back on the queue
                    self.results.put_nowait(next_result)
        except Empty:
            return

    def run(self) -> None:
        """
        This is the function that the process will run.
        First instantiate the diffusion pipeline, then communicate as needed.
        """
        from pibble.util.helpers import OutputCatcher
        from pibble.util.log import ConfigurationLoggingContext

        catcher = OutputCatcher()

        with ConfigurationLoggingContext(self.configuration, prefix="enfugue.engine.logging."):
            with catcher:
                last_data = datetime.datetime.now()
                idle_seconds = 0.0

                while True:
                    try:
                        payload = self.instructions.get(timeout=self.POLLING_DELAY_MS / 1000)
                    except KeyboardInterrupt:
                        return
                    except Empty:
                        idle_seconds = (datetime.datetime.now() - last_data).total_seconds()
                        if idle_seconds > self.idle_seconds:
                            logger.info(
                                f"Reached maximum idle time after {idle_seconds:.1f} seconds, exiting engine process"
                            )
                            return
                        continue
                    except Exception as ex:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(traceback.format_exc())
                        raise IOError("Received unexpected {0}, process will exit. {1}".format(type(ex).__name__, ex))

                    instruction = Serializer.deserialize(payload)
                    if not isinstance(instruction, dict):
                        logger.error(f"Unexpected non-dictionary argument {instruction}")
                        continue

                    instruction_id = instruction["id"]
                    instruction_action = instruction["action"]
                    instruction_payload = instruction.get("payload", None)

                    logger.debug(f"Received instruction {instruction_id}, action {instruction_action}")
                    if instruction_action == "ping":
                        logger.debug("Responding with 'pong'")
                        self.results.put(Serializer.serialize({"id": instruction_id, "result": "pong"}))
                    elif instruction_action in ["exit", "stop"]:
                        logger.debug("Exiting process")
                        self.pipemanager.unload_inpainter("exiting")
                        self.pipemanager.unload_refiner("exiting")
                        self.pipemanager.unload_pipeline("exiting")
                        return
                    elif instruction_action in ["invoke", "plan"]:
                        response = {"id": instruction_id, "payload": instruction_payload}
                        try:
                            if instruction_action == "plan":
                                intermediate_dir = instruction_payload.get("intermediate_dir", None)
                                intermediate_steps = instruction_payload.get("intermediate_steps", None)
                                plan = self.get_diffusion_plan(instruction_payload)
                                response["result"] = self.execute_diffusion_plan(
                                    instruction_id,
                                    plan,
                                    intermediate_dir=intermediate_dir,
                                    intermediate_steps=intermediate_steps,
                                )
                            else:
                                payload = self.check_invoke_kwargs(instruction_id, **instruction_payload)
                                response["result"] = self.pipemanager(**payload)
                        except Exception as ex:
                            response["error"] = qualify(type(ex))
                            response["message"] = str(ex)
                            
                            # Also log so this appears in the engine log
                            logger.error(f"Received error {response['error']}: {response['message']}")
                            if logger.isEnabledFor(logging.DEBUG):
                                response["trace"] = traceback.format_exc()
                                logger.debug(response["trace"])

                        del self.pipemanager.keepalive_callback
                        self.results.put(Serializer.serialize(response))
                        self.clear_intermediates(instruction_id)
                    else:
                        self.results.put(
                            Serializer.serialize(
                                {
                                    "id": instruction_id,
                                    "error": f"Unknown action '{instruction_action}'",
                                }
                            )
                        )
                    out, err = catcher.output()
                    if out:
                        logger.debug(f"stdout: {out}")
                    if err:
                        logger.error(f"stderr: {err}")
                    catcher.clean()
                    last_data = datetime.datetime.now()
