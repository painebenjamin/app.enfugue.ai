import time
import threading
import requests

from pibble.api.configuration import APIConfiguration


class OpenBrowserWhenResponsiveThread(threading.Thread):
    """
    This thread will open a browser once it receives a positive response
    """

    POLLING_INTERVAL = 1

    def __init__(self, configuration: APIConfiguration) -> None:
        super(OpenBrowserWhenResponsiveThread, self).__init__()
        server_config = configuration.get("server", {})
        scheme = "https" if server_config.get("secure", False) else "http"
        host = server_config.get("domain", "127.0.0.1")
        port = server_config.get("port", 45554)
        self.url = f"{scheme}://{host}:{port}/"
        self.stop_event = threading.Event()

    @property
    def stopped(self) -> bool:
        """
        Returns whether or not the stop event is set.
        """
        return self.stop_event.is_set()

    def stop(self) -> None:
        """
        Sets the stop event.
        """
        self.stop_event.set()

    def run(self) -> None:
        """
        Waits for responsiveness then opens a browser.
        """
        while not self.stopped:
            time.sleep(self.POLLING_INTERVAL)
            try:
                r = requests.get(self.url)
                if 200 <= r.status_code < 400:
                    import webbrowser

                    webbrowser.open(self.url)
                    self.stop()
            except:
                pass
