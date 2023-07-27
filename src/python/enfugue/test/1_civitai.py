"""
Performs a simple unit test of CivitAI functionality
"""
from enfugue.partner import CivitAI
from enfugue.util import logger
import json
from pibble.util.log import DebugUnifiedLoggingContext
from pibble.util.helpers import Assertion

def main() -> None:
    with DebugUnifiedLoggingContext():
        civitai = CivitAI()
        civitai.configure()
        i = 0
        for i, model in enumerate(
            civitai.get_models(
                types = "Checkpoint",
                sort = "Most Downloaded",
                period = "Month",
                limit = 10
            )
        ):
            logger.debug(model)
        Assertion(Assertion.GT)(i, 0)

if __name__ == "__main__":
    main()
