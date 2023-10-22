import os
from typing import List, Dict, Any

from webob import Request, Response

from pibble.api.exceptions import BadRequestError, StateConflictError
from pibble.ext.user.server.base import UserExtensionHandlerRegistry

from enfugue.util import check_make_directory
from enfugue.partner.civitai import CivitAI
from enfugue.api.controller.base import EnfugueAPIControllerBase

__all__ = ["EnfugueAPIDownloadsController"]


class EnfugueAPIDownloadsController(EnfugueAPIControllerBase):
    handlers = UserExtensionHandlerRegistry()

    @property
    def civitai(self) -> CivitAI:
        """
        Gets the client. Instantiates if necessary.
        """
        if not hasattr(self, "_civitai"):
            self._civitai = CivitAI()
            self._civitai.configure()
        return self._civitai

    @handlers.path("^/api/download$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("Download", "create")
    def start_download(self, request: Request, response: Response) -> Dict[str, Any]:
        """
        Starts a download.
        """
        try:
            download_type = str(request.parsed["type"])
            target_dir = self.get_configured_directory(download_type)
            target_file = os.path.join(target_dir, request.parsed["filename"])

            if os.path.exists(target_file) and not request.parsed.get("overwrite", False):
                raise StateConflictError(f"File exists: {request.parsed['filename']}")

            check_make_directory(target_dir)

            return self.manager.download(request.token.user.id, request.parsed["url"], target_file).format()
        except KeyError as ex:
            raise BadRequestError(f"Missing required argument {ex}")
    
    @handlers.path("^/api/download/cancel$")
    @handlers.methods("POST")
    @handlers.format()
    @handlers.secured("Download", "delete")
    def cancel_download(self, request: Request, response: Response) -> None:
        """
        Cancels a download.
        """
        try:
            if self.manager.cancel_download(request.parsed["url"]):
                return
            raise BadRequestError(f"Not downloading {request.parsed['url']}")
        except KeyError as ex:
            raise BadRequestError(f"Missing required argument {ex}")

    @handlers.path("^/api/download$")
    @handlers.methods("GET")
    @handlers.secured()
    @handlers.format()
    def get_download_status(self, request: Request, response: Response) -> List[Dict[str, Any]]:
        """
        Gets status for downloads. Tese are only stored in memory.
        """
        return [download.format() for download in self.manager.get_downloads(request.token.user.id)]

    @handlers.path("^/api/civitai/(?P<lookup>[a-z]+)$")
    @handlers.methods("GET")
    @handlers.secured()
    @handlers.format()
    def civitai_lookup(self, request: Request, response: Response, lookup: str) -> List[Dict[str, Any]]:
        """
        Performs a lookup in CivitAI
        """
        lookup_type = {
            "checkpoint": "Checkpoint",
            "inversion": "TextualInversion",
            "lora": "LORA",
            "lycoris": "LoCon",
            "controlnet": "Controlnet",
            "poses": "Poses",
            "hypernetwork": "Hypetnetwork",
            "gradient": "AestheticGradient",
            "motion": "MotionModule",
        }.get(lookup, None)

        if lookup_type is None:
            raise BadRequestError(f"Unknown lookup type {lookup}")

        query = request.params.get("query", None)
        show_nsfw = request.params.get("nsfw", False)
        if isinstance(show_nsfw, str):
            show_nsfw = show_nsfw.lower() in ["t", "true", "y", "yes", "1"]
        else:
            show_nsfw = bool(show_nsfw)

        lookup_kwargs = {
            "types": lookup_type,
            "limit": 20,
            "nsfw": False if self.safe else show_nsfw,
            "query": request.params.get("query", None),
            "page": request.params.get("page", None),
            "sort": request.params.get("sort", "Most Downloaded"),
            "period": request.params.get("period", "Month"),
            "allowCommercialUse": request.params.get("allow_commercial_use", None),
        }

        iterator = self.civitai.get_models(**lookup_kwargs)  # type: ignore
        return [result.serialize() for result in iterator]
