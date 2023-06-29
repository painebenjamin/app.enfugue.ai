import os
import tempfile
from typing import Dict, Any

from enfugue.api import EnfugueAPIServer
from enfugue.interface import EnfugueInterfaceServer
from enfugue.util import get_signature, logger

__all__ = ["EnfugueSecureServer", "EnfugueServer"]


class EnfugueSecureServer(EnfugueAPIServer):
    """
    This class tries to get signing details at configuration time.
    """

    def configure(self, server=Dict[str, Any], **kwargs: Any) -> None:
        """
        At configuration, get cached or remote resources.
        """
        secure = server.get("secure", True)
        domain = server.get("domain", True)
        if secure and domain == "app.enfugue.ai":
            try:
                key, cert, chain = get_signature()
                directory = tempfile.mkdtemp()
                keyfile = os.path.join(directory, "key.pem")
                certfile = os.path.join(directory, "cert.pem")
                chainfile = os.path.join(directory, "chain.pem")
                open(keyfile, "w").write(key)
                open(certfile, "w").write(cert)
                open(chainfile, "w").write(chain)
                server["cert"] = certfile
                server["key"] = keyfile
                server["chain"] = chainfile
            except Exception as ex:
                logger.error(f"Couldn't get signatures, disabling SSL. {ex}")
                server["secure"] = False
        super(EnfugueSecureServer, self).configure(server=server, **kwargs)

    def on_destroy(self) -> None:
        """
        On destroy, clear tempfiles.
        """
        if hasattr(self, "_keyfile"):
            try:
                os.remove(self._keyfile)
            except:
                pass
        if hasattr(self, "_certfile"):
            try:
                os.remove(self._certfile)
            except:
                pass


class EnfugueServer(EnfugueSecureServer, EnfugueInterfaceServer):
    pass
