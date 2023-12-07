from __future__ import annotations

from typing import Iterator, Callable, Optional, Dict, List, Any

from pibble.api.server.webservice.jsonapi import JSONWebServiceAPIServer

from enfugue.api.invocations import InvocationMonitor
from enfugue.api.downloads import Download
from enfugue.api.manager import SystemManager

from enfugue.partner.civitai import CivitAI

__all__ = ["EnfugueAPIControllerBase"]


class EnfugueAPIControllerBase(JSONWebServiceAPIServer):
    # Properties defined in server
    default_checkpoints: Dict[str, str]
    default_lora: Dict[str, str]
    engine_root: str
    safe: bool
    thumbnail_height: int
    civitai: CivitAI
    manager: SystemManager

    # Methods defined in server
    invoke: Callable[..., InvocationMonitor]
    download: Callable[[int, str, str], Download]
    check_name: Callable[[str], None]
    check_find_model: Callable[[str, str], str]
    get_checksum: Callable[[str], str]
    get_civitai_metadata: Callable[[str], List[Dict[str, Any]]]
    get_models_in_directory: Callable[[str], List[str]]
    get_model_metadata: Callable[[str, List[str]], Optional[Dict[str, Any]]]
    get_diffusers_models: Callable[[], List[str]]
    get_configured_directory: Callable[[str], str]
    get_default_model: Callable[[str], Optional[str]]
    get_plan_kwargs_from_model: Callable[..., Dict[str, Any]]
    get_default_size_for_model: Callable[[str], int]

    @staticmethod
    def enumerate() -> Iterator[EnfugueAPIControllerBase]:
        """
        Iterates through declared subcontroller classes.
        """
        for cls in EnfugueAPIControllerBase.__subclasses__():
            yield cls
