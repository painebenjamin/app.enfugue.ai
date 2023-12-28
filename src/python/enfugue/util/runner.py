from __future__ import annotations

import os
import time

from typing import Dict, Any, Optional, Callable, List
from copy import deepcopy
from threading import Event

from pibble.util.helpers import OutputCatcher
from pibble.api.exceptions import ConfigurationError

from enfugue.util import logger, get_local_static_directory
from enfugue.util.browser import OpenBrowserWhenResponsiveThread

__all__ = ["EnfugueServerRunner"]

class EnfugueServerRunner:
    """
    Runs one or many servers
    """
    def __init__(self, configuration: Dict[str, Any]) -> None:
        self.configuration = configuration

    @property
    def url(self) -> str:
        """
        Gets the first configured URL
        """
        secure = self.configuration["server"].get("secure", False)
        if isinstance(secure, list) or isinstance(secure, tuple):
            secure = secure[0]
        domain = self.configuration["server"].get("domain", "127.0.0.1")
        if isinstance(domain, list) or isinstance(domain, tuple):
            domain = domain[0]
        port = self.configuration["server"].get("port", "127.0.0.1")
        if isinstance(port, list) or isinstance(port, tuple):
            port = port[0]
        scheme = "https" if secure else "http"
        return f"{scheme}://{domain}:{port}/"

    def run(
        self,
        url_callback: Optional[Callable[[str], None]] = None,
        until: Optional[Event] = None,
        open_browser: bool = False
    ) -> None:
        """
        Runs the servers (blocking)
        """
        from enfugue.server import EnfugueServer
        # Determine how many servers we need to run
        try:
            all_host = self.configuration["server"]["host"]
            all_port = self.configuration["server"]["port"]
            all_domain = self.configuration["server"].get("domain", None)
            all_secure = self.configuration["server"].get("secure", False)
            all_cert = self.configuration["server"].get("cert", None)
            all_key = self.configuration["server"].get("key", None)
        except KeyError as ex:
            logger.warning(f"Couldn't find {ex} in {self.configuration}")
            raise ConfigurationError(f"Missing required configuration key {ex}")

        num_servers = 1
        for var in [all_host, all_secure, all_domain, all_port, all_cert, all_key]:
            if isinstance(var, list) or isinstance(var, tuple):
                num_servers = max(num_servers, len(var))

        def get_for_index(index: int, config_value: Any) -> Any:
            """
            Gets a configuration value for an index, or the last, or the only
            """
            if isinstance(config_value, list) or isinstance(config_value, tuple):
                if index < len(config_value):
                    return config_value[index]
                return config_value[-1]
            return config_value

        servers: List[EnfugueServer] = []
        open_browser_thread: Optional[OpenBrowserWhenResponsiveThread] = None

        try:
            if open_browser:
                open_browser_thread = OpenBrowserWhenResponsiveThread(self.url)
                open_browser_thread.start()

            for i in range(num_servers):
                host = get_for_index(i, all_host)
                port = get_for_index(i, all_port)
                secure = get_for_index(i, all_secure)
                cert = get_for_index(i, all_cert)
                key = get_for_index(i, all_key)
                domain = get_for_index(i, all_domain)

                this_configuration = deepcopy(self.configuration)
                this_configuration["server"]["host"] = host
                this_configuration["server"]["port"] = port
                this_configuration["server"]["domain"] = domain
                this_configuration["server"]["secure"] = secure
                this_configuration["server"]["cert"] = cert
                this_configuration["server"]["key"] = key

                server = EnfugueServer()

                scheme = "https" if secure else "http"
                port_echo = "" if port in [80, 443] else f":{port}"
                domain_echo = "127.0.0.1" if not domain else domain
                url = f"{scheme}://{domain_echo}{port_echo}/"
                if num_servers == 1:
                    server.configure(**this_configuration)
                    if url_callback:
                        url_callback(url)
                    logger.info(f"Running enfugue at {scheme}://{domain_echo}{port_echo}/")
                    server.serve()
                    return
                if url_callback:
                    url_callback(url)
                logger.info(f"Enfugue server {(i+1):d} of {num_servers:d} listening on {scheme}://{domain_echo}{port_echo}/")
                server.configure_start(**this_configuration)
                servers.append(server)
            if until is not None:
                while not until.is_set():
                    until.wait(1)
            else:
                while True:
                    time.sleep(1)
        finally:
            if open_browser_thread is not None:
                try:
                    open_browser_thread.stop()
                    open_browser_thread.join()
                except:
                    pass
            for server in servers:
                try:
                    server.stop()
                except:
                    pass

    def run_icon(
        self,
        url_callback: Optional[Callable[[str], None]] = None,
        open_browser: bool = False
    ) -> None:
        """
        When working on windows using a .exe, this runs the server using a system tray icon
        """
        import pystray
        from PIL import Image

        stop_event = Event()

        def stop(icon: pystray.Icon) -> None:
            """
            Stops the server.
            """
            icon.visible = False
            stop_event.set()
            icon.stop()

        def open_app(icon: pystray.Icon) -> None:
            """
            Opens up a browser and navigates to the app
            """
            import webbrowser
            webbrowser.open(self.url)

        def setup(icon: pystray.Icon) -> None:
            """
            Starts the server.
            """
            icon.visible = True
            self.run(until=stop_event, url_callback=url_callback, open_browser=open_browser)

        static_dir = get_local_static_directory()
        icon_path = self.configuration.get("enfugue", {}).get("icon", "favicon/favicon-64x64.png")
        icon_image = Image.open(os.path.join(static_dir, "img", icon_path))
        icon = pystray.Icon("enfugue", icon_image)
        icon.menu = pystray.Menu(pystray.MenuItem("Open App", open_app), pystray.MenuItem("Quit", stop))

        catcher = OutputCatcher()
        with catcher:
            icon.run(setup=setup)
