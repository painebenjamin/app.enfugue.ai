import os
from typing import List, Dict, Any

from webob import Request, Response

from pibble.api.exceptions import BadRequestError, StateConflictError
from pibble.ext.user.server.base import UserExtensionHandlerRegistry


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
            target_dir = os.path.join(self.engine_root, str(request.parsed["type"]))
            target_file = os.path.join(target_dir, request.parsed["filename"])

            if os.path.exists(target_file) and not request.parsed.get("overwrite", False):
                raise StateConflictError(f"File exists: {request.parsed['filename']}")

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            return self.manager.download(
                request.token.user.id, request.parsed["url"], target_file
            ).format()
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
    def civitai_lookup(
        self, request: Request, response: Response, lookup: str
    ) -> List[Dict[str, Any]]:
        """
        Performs a lookup in CivitAI
        """
        lookup_type = {
            "checkpoint": "Checkpoint",
            "inversion": "TextualInversion",
            "lora": "LORA",
            "controlnet": "Controlnet",
            "poses": "Poses",
            "hypernetwork": "Hypetnetwork",
            "gradient": "AestheticGradient",
        }.get(lookup, None)

        if lookup_type is None:
            raise BadRequestError(f"Unknown lookup type {lookup_type}")

        query = request.params.get("query", None)

        lookup_kwargs = {
            "types": lookup_type,
            "limit": 20,
            "nsfw": not self.safe,
            "query": request.params.get("query", None),
            "page": request.params.get("page", None),
            "sort": request.params.get("sort", "Most Downloaded"),
            "period": request.params.get("period", "Month"),
            "allowCommercialUse": request.params.get("allow_commercial_use", None),
        }

        iterator = self.civitai.get_models(**lookup_kwargs)  # type: ignore
        return [result.serialize() for result in iterator]
