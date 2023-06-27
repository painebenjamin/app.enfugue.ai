import os
import base64
import datetime
import requests

from typing import List

from enfugue.util.log import logger
from enfugue.util.installation import get_local_config_directory

MAX_SIGNATURE_AGE = 60 * 60 * 24 * 30


__all__ = ["get_signature"]


def get_signature() -> List[str]:
    """
    Gets necessary signing details for the webserver
    """
    from enfugue.util.security import decrypt  # type: ignore[attr-defined]

    local = os.path.join(get_local_config_directory(), "payload.txt")
    contents = None

    if os.path.exists(local):
        saved = datetime.datetime.fromtimestamp(os.path.getmtime(local))
        if (datetime.datetime.now() - saved).total_seconds() > MAX_SIGNATURE_AGE:
            logger.info("Expiring local signatures.")
            os.remove(local)
        else:
            contents = open(local, "rb").read()
    if contents is None:
        logger.info("Fetching remote signatures.")
        contents = requests.get("https://cdn.enfugue.ai/payload.txt").content
        open(local, "wb").write(contents)

    return [
        base64.b64decode(decrypt(chunk)).decode("utf-8")
        for chunk in base64.b64decode(contents).decode("utf-8").split(":")
    ]
