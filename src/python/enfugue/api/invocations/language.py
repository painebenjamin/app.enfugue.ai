from __future__ import annotations

from datetime import datetime

from enfugue.api.invocations.base import InvocationMonitor

from typing import Optional, List, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from enfugue.diffusion.engine import DiffusionEngine

__all__ = ["CaptionInvocationMonitor"]


class CaptionInvocationMonitor(InvocationMonitor):
    """
    Holds the details for a single invocation
    """
    last_captions: Optional[List[str]] = None

    def __init__(
        self,
        engine: DiffusionEngine,
        communication_timeout: Optional[int] = 180,
        **kwargs: Any,
    ) -> None:
        super(CaptionInvocationMonitor, self).__init__(
            engine,
            communcation_timeout=communication_timeout,
        )
        self.plan = kwargs["plan"]

    def start(self) -> None:
        """
        Starts the invocation (locks)
        """
        with self.lock:
            self.start_time = datetime.now()
            payload: Dict[str, Any] = {}
            payload["plan"] = self.plan
            self.id = self.engine.dispatch("language", payload)

    def format(self) -> Dict[str, Any]:
        """
        Formats the invocation to a dictionary
        """
        formatted = super(CaptionInvocationMonitor, self).format()
        if formatted["status"] == "processing":
            formatted["captions"] = self.last_captions if self.last_captions is not None else None
        return formatted
