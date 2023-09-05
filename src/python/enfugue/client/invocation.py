from __future__ import annotations

import io
import PIL
import time

from typing import Literal, Dict, List, Any, TYPE_CHECKING

from enfugue.util import logger

if TYPE_CHECKING:
    from enfugue.client.client import EnfugueClient

__all__ = ["RemoteInvocation"]


class RemoteInvocation:
    """
    Represents an invocation of the engine, which can be tracked
    synchronously or asynchronously.
    """

    def __init__(
        self, client: EnfugueClient, uuid: str, status: Literal["queued", "processing", "error", "completed"]
    ) -> None:
        self.client = client
        self.uuid = uuid
        self.status = status

    @staticmethod
    def from_response(client: EnfugueClient, response: Dict[str, Any]) -> RemoteInvocation:
        """
        Parses UUID from the response.
        """
        try:
            uuid = response["uuid"]
            status = response["status"]
            return RemoteInvocation(client, uuid, status)
        except KeyError:
            raise RuntimeError(f"Unparseable response from the server: {response}")

    def get_status(self) -> Dict[str, Any]:
        """
        Gets the current status of the invocation
        """
        return self.client.get(f"/invocation/{self.uuid}").json().get("data", {})

    def delete(self) -> None:
        """
        Deletes the invocation and results.
        """
        self.client.delete(f"/invocation/{self.uuid}")

    def results(self, polling_interval: int = 5) -> List[PIL.Image.Image]:
        """
        Parses results from a successful invocation. Waits for it to complete.
        """
        try:
            status = self.get_status()
            while status["status"] in ["queued", "processing"]:
                logger.debug(f"Invocation not complete yet, checking again in {polling_interval}")
                time.sleep(polling_interval)
                status = self.get_status()
            if status["status"] == "error":
                raise RuntimeError(status.get("message", "The remote server did not include an error message."))
            duration = status["duration"]
            logger.info(f"Invocation complete in {duration:.2f}")
            images = []
            response_images = status.get("images", [])
            if response_images:
                for image in response_images:
                    image_bytes = self.client.get(f"/invocation/{image}", stream=True).content
                    images.append(PIL.Image.open(io.BytesIO(image_bytes)))
            return images
        except KeyError:
            raise RuntimeError("Unparseable response sent from the server. Check logs for details.")
