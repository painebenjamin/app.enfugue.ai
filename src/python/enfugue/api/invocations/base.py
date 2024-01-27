from __future__ import annotations

from typing import Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from enfugue.diffusion.engine import DiffusionEngine

from datetime import datetime
from multiprocessing import Lock

from pibble.util.strings import get_uuid


__all__ = ["InvocationMonitor"]


class TerminatedError(Exception):
    pass

class InvocationMonitor:
    """
    Holds the details for an invocation
    """
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_intermediate_time: Optional[datetime] = None
    id: Optional[int] = None
    last_step: Optional[int] = None
    last_total: Optional[int] = None
    last_rate: Optional[float] = None
    last_task: Optional[str] = None
    error: Optional[Exception] = None
    result: Any = None

    def __init__(
        self,
        engine: DiffusionEngine,
        communication_timeout: Optional[int]=180,
        **kwargs: Any
    ) -> None:
        self.lock = Lock()
        self.uuid = get_uuid()
        self.engine = engine
        self.communication_timeout = communication_timeout

    def _communicate(self) -> None:
        """
        Tries to communicate with the engine to see what's going on.
        """
        if self.id is None:
            raise IOError("Invocation not started yet.")
        if self.result is not None:
            raise IOError("Invocation already completed.")
        try:
            start_comm = datetime.now()
            last_intermediate = self.engine.last_intermediate(self.id)
            if last_intermediate is not None:
                for key in last_intermediate:
                    setattr(self, f"last_{key}", last_intermediate[key])
                self.last_intermediate_time = datetime.now()
            end_comm = (datetime.now() - start_comm).total_seconds()

            try:
                result = self.engine.wait(self.id, timeout=0.1)
            except TimeoutError:
                raise
            except Exception as ex:
                result = None
                self.error = ex

            if result is not None:
                # Complete
                self.result = result
                self.end_time = datetime.now()

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
        raise NotImplementedError()

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
        if self.end_time is not None and self.result is not None or self.error is not None:
            return False
        if self.communication_timeout is None:
            return False
        if self.last_intermediate_time is not None:
            last_known_time = self.last_intermediate_time
        else:
            last_known_time = self.start_time # type: ignore[unreachable]
        seconds_since_last_communication = (datetime.now() - last_known_time).total_seconds()
        return seconds_since_last_communication > self.communication_timeout

    def timeout(self) -> None:
        """
        Times out an invocation that got lost
        """
        self.error = TimeoutError("Invocation timed out.")
        self.end_time = datetime.now()

    def terminate(self) -> None:
        """
        Kills an active invocation
        """
        if self.id is None:
            raise IOError("Invocation not started yet.")
        if self.result is not None:
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
                    "result": self.result,
                }

            try:
                if self.result is not None:
                    status = "completed"
                    result = self.result
                else:
                    status = "processing"
                    result = None
                    self._communicate()
                    if self.result is not None:
                        # Finished in previous _communicate() calling
                        status = "completed" # type: ignore[unreachable]
                        result = self.result

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

            formatted = {
                "type": type(self).__name__,
                "id": self.id,
                "uuid": self.uuid,
                "status": status,
                "progress": progress,
                "step": step,
                "duration": duration,
                "total": total,
                "rate": rate,
                "task": self.last_task,
                "result": result
            }

            return formatted

    def __str__(self) -> str:
        """
        Stringifies the invocation for debugging.
        """

        return f"Invocation {self.uuid}, last step: {self.last_step}, last total: {self.last_total}: error: {self.error}, result: {self.result}"
