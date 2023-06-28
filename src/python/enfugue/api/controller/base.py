from __future__ import annotations

import os

from typing import Iterator, Callable

from pibble.api.server.webservice.jsonapi import JSONWebServiceAPIServer
from enfugue.util import check_make_directory
from enfugue.api.invocations import Invocation
from enfugue.api.downloads import Download
from enfugue.api.manager import SystemManager

__all__ = ["EnfugueAPIControllerBase"]


class EnfugueAPIControllerBase(JSONWebServiceAPIServer):
    invoke: Callable[..., Invocation]
    download: Callable[[int, str, str], Download]
    manager: SystemManager

    @property
    def safe(self) -> bool:
        """
        Returns whether or not to use safety filters.
        """
        return self.configuration.get("enfugue.safe", True)

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

    @staticmethod
    def enumerate() -> Iterator[EnfugueAPIControllerBase]:
        """
        Iterates through declared subcontroller classes.
        """
        for cls in EnfugueAPIControllerBase.__subclasses__():
            yield cls
