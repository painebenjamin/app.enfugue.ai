from __future__ import annotations

import time
import datetime

from typing import Optional, Any, Union, Dict, Type, List, TYPE_CHECKING
from typing_extensions import Self
from multiprocessing import Queue as MakeQueue
from multiprocessing.queues import Queue
from queue import Empty

from pibble.api.configuration import APIConfiguration
from pibble.util.strings import Serializer
from pibble.util.helpers import resolve

from enfugue.util import logger
from enfugue.diffusion.process import (
    EngineProcess,
    DiffusionEngineProcess
)

if TYPE_CHECKING:
    from enfugue.diffusion.invocation import LayeredInvocation

__all__ = [
    "Engine",
    "DiffusionEngine"
]

class Engine:
    LOGGING_DELAY_MS = 10000
    process: EngineProcess

    def __init__(self, configuration: Optional[APIConfiguration] = None):
        self.configuration = APIConfiguration()
        self.request = 0

        if configuration is not None:
            self.configuration = configuration

    @property
    def process_class(self) -> Type[EngineProcess]:
        """
        Gets the class of the process
        """
        return EngineProcess

    def __enter__(self) -> Self:
        """
        When entering the engine via context manager, start the process.
        """
        self.spawn_process(fail_when_redundant=False)
        return self

    def __exit__(self, *args: Any) -> None:
        """
        When exiting the engine via context manager, kill the process.
        """
        self.terminate_process()

    def check_get_queue(self, queue_name: str) -> Queue:
        """
        Gets a queue, creates it if not already made.
        """
        if not hasattr(self, f"_{queue_name}"):
            if self.is_alive():
                raise IOError(f"Queue '{queue_name}' is missing, but process is alive.")
            setattr(self, f"_{queue_name}", MakeQueue())
        return getattr(self, f"_{queue_name}")

    def check_delete_queue(self, queue_name: str) -> None:
        """
        Deletes a queue.
        """
        if hasattr(self, f"_{queue_name}"):
            delattr(self, f"_{queue_name}")

    @property
    def instructions(self) -> Queue:
        """
        Gets the instructions queue
        """
        return self.check_get_queue("instructions")

    @instructions.deleter
    def instructions(self) -> None:
        """
        Deletes the instructions queue
        """
        self.check_delete_queue("instructions")

    @property
    def intermediates(self) -> Queue:
        """
        Gets the intemerdiates queue
        """
        return self.check_get_queue("intermediates")

    @intermediates.deleter
    def intermediates(self) -> None:
        """
        Deletes the intermediates queue
        """
        self.check_delete_queue("intermediates")

    @property
    def results(self) -> Queue:
        """
        Gets the results queue
        """
        return self.check_get_queue("results")

    @results.deleter
    def results(self) -> None:
        """
        Deletes the results queue
        """
        self.check_delete_queue("results")

    def get_queues(self) -> List[Queue]:
        """
        Gets queues to pass to the process
        """
        return [self.instructions, self.results, self.intermediates]

    def delete_queues(self) -> None:
        """
        Deletes queues after killing a process
        """
        try:
            del self.instructions
        except:
            pass
        try:
            del self.results
        except:
            pass
        try:
            del self.intermediates
        except:
            pass

    def spawn_process(
        self,
        timeout: Optional[Union[int, float]] = None,
        fail_when_redundant: bool = True,
    ) -> None:
        """
        Starts the engine process and waits for a response
        """
        if hasattr(self, "process") and self.process.is_alive():
            if fail_when_redundant:
                raise IOError("spawn_process called while a process is already running")
            return

        logger.debug("No current engine process, creating.")
        poll_delay_seconds = self.process_class.POLLING_DELAY_MS / 250
        self.process = self.process_class(self.configuration, *self.get_queues())

        try:
            logger.debug("Starting process.")
            self.process.start()
            time.sleep(poll_delay_seconds)
            if not self.is_alive():
                raise IOError("Engine process died before it became responsive")
        except Exception as ex:
            self.terminate_process()
            raise ex

    def is_alive(self, timeout: Optional[Union[int, float]] = None) -> bool:
        """
        Determines if the process is alive.
        """
        if not hasattr(self, "process"):
            return False

        if not self.process.is_alive():
            self.terminate_process()
            return False

        return True

    def terminate_process(self, timeout: Optional[Union[int, float]] = 10) -> None:
        """
        Stops the process if it is running.
        """
        if not hasattr(self, "process"):
            return

        if self.process.is_alive():
            start = datetime.datetime.now()
            sleep_time = self.process_class.POLLING_DELAY_MS / 500
            self.dispatch("stop")
            time.sleep(sleep_time)
            while self.process.is_alive():
                waited = (datetime.datetime.now() - start).total_seconds()
                if timeout is not None and timeout > waited:
                    logger.debug("Process did not stop on it's own, sending TERM")
                    self.process.terminate()
                    time.sleep(sleep_time)
                    break
                time.sleep(sleep_time)

        if hasattr(self, "process") and self.process.is_alive():
            logger.debug("Sending term one more time...")
            self.process.terminate()
            time.sleep(sleep_time)

        if hasattr(self, "process"):
            if self.process.is_alive():
                raise IOError("Couldn't terminate process")
            delattr(self, "process")
        self.delete_queues()

    def keepalive(self, timeout: Union[int, float] = 0.2) -> bool:
        """
        Keeps the process alive if it isn't already.
        """
        if not hasattr(self, "process"):
            return False

        if not self.process.is_alive():
            self.terminate_process()
            return False

        if self.instructions.empty():
            ping_response = self.invoke("ping")
            if ping_response != "pong":
                self.terminate_process()
                raise IOError(f"Incorrect ping response {ping_response}")
        return True

    def dispatch(
        self,
        action: str,
        payload: Any = None,
        spawn_process: bool = True
    ) -> Any:
        """
        Sends a payload, does not wait for a response.
        """
        if spawn_process:
            self.spawn_process(timeout=1, fail_when_redundant=False)
        self.request += 1
        envelope = {"id": self.request, "action": action, "payload": payload}
        logger.debug(f"Dispatching envelope {envelope} to {self.instructions}")
        self.instructions.put(Serializer.serialize(envelope))
        return envelope["id"]

    def last_intermediate(self, id: int) -> Any:
        """
        Gets the last steps' details, if any
        """
        all_steps = []

        try:
            while True:
                all_steps.append(self.intermediates.get_nowait())
        except Empty:
            pass

        intermediate_data: Optional[Dict[str, Any]] = None
        for step in all_steps:
            step_deserialized = Serializer.deserialize(step)
            if step_deserialized["id"] == id:
                if intermediate_data is None:
                    intermediate_data = {"id": id}
                for key in ["step", "total", "rate", "images", "task", "video"]:
                    if key in step_deserialized:
                        intermediate_data[key] = step_deserialized[key]
            else:
                # Wrong ID, put back in queue
                self.intermediates.put_nowait(step)
        return intermediate_data

    def wait(self, id: int, timeout: Optional[Union[int, float]] = None) -> Any:
        """
        Waits for a response after issuing a request and getting an ID
        """
        start = datetime.datetime.now()
        poll_delay_seconds = DiffusionEngineProcess.POLLING_DELAY_MS / 500

        while True:
            if not self.is_alive():
                raise IOError("Process died while waiting for result.")
            to_raise, to_return = None, None
            all_results = []
            try:
                while True:
                    all_results.append(self.results.get_nowait())
            except Empty:
                pass

            for result in all_results:
                result_deserialized = Serializer.deserialize(result)
                try:
                    if result_deserialized["id"] == id:
                        if "error" in result_deserialized:
                            to_raise = resolve(result_deserialized["error"])(result_deserialized["message"])
                            if "trace" in result_deserialized:
                                logger.error(result_deserialized["trace"])
                        else:
                            to_return = result_deserialized["result"]
                    else:
                        self.results.put_nowait(result)
                except:
                    logger.error(f"Couldn't parse result {result}")
                    raise
            if to_raise is not None:
                raise to_raise
            if to_return is not None:
                return to_return

            time.sleep(poll_delay_seconds)
            if timeout is not None and (datetime.datetime.now() - start).total_seconds() > timeout:
                raise TimeoutError("Timed out waiting for response.")

    def invoke(
        self,
        action: str,
        payload: Any = None,
        timeout: Optional[Union[int, float]] = None
    ) -> Any:
        """
        Issue a single request synchronously using arg syntax.
        """
        return self.wait(self.dispatch(action, payload), timeout)

class DiffusionEngine(Engine):
    """
    The base extension of this class is for the stable diffusion engine
    """
    @property
    def process_class(self) -> Type[EngineProcess]:
        """
        Change to diffusionengine
        """
        return DiffusionEngineProcess

    def execute(
        self,
        plan: LayeredInvocation,
        timeout: Optional[Union[int, float]] = None, wait: bool = False
    ) -> Any:
        """
        This is a helpful method to just serialize and execute a plan.
        """
        id = self.dispatch("plan", plan.serialize())
        if wait:
            return self.wait(id, timeout)
        return id

    def __call__(self, timeout: Optional[Union[int, float]] = None, **kwargs: Any) -> Any:
        """
        Issues a single invocation request using kwarg syntax.
        """
        return self.wait(self.dispatch("invoke", kwargs), timeout)

    @staticmethod
    def debug() -> DiffusionEngine:
        """
        Gets a DiffusionEngine instance with debug logging enabled.
        """
        return DiffusionEngine(
            APIConfiguration(
                enfugue={
                    "engine": {
                        "logging": {
                            "level": "DEBUG",
                            "handler": "stream",
                            "stream": "stdout",
                            "colored": True,
                        }
                    }
                }
            )
        )
