from pibble.api.server.webservice.base import WebServiceAPIServerBase
from pibble.util.log import DebugUnifiedLoggingContext as ctx

with ctx():
    server = WebServiceAPIServerBase()
    server.configure(
        server={
            "host": "0.0.0.0",
            "port": 1246,
            "driver": "cherrypy"
        }
    )
    try:
        server.serve()
    except Exception as ex:
        print(f"LEAVING ON {ex}")
