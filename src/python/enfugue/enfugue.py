"""
This is the entry point for the executable
"""
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # pyinstaller fix

    from enfugue.util import get_local_configuration
    import os
    import certifi
    import platform

    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()

    from enfugue.server import EnfugueServer
    from enfugue.util.browser import OpenBrowserWhenResponsiveThread
    from typing import Optional

    system = platform.system()
    configuration = get_local_configuration()
    open_browser_thread: Optional[OpenBrowserWhenResponsiveThread] = None
    server = None

    open_browser: bool = configuration.get("open", os.getenv("ENFUGUE_OPEN", True))
    if isinstance(open_browser, str): # type: ignore
        open_browser = open_browser[0].lower() in ["1", "t", "y"] # type: ignore[unreachable]

    if open_browser:
        open_browser_thread = OpenBrowserWhenResponsiveThread(configuration)
    try:
        if open_browser_thread is not None:
            open_browser_thread.start()
        if system == "Windows":
            EnfugueServer.serve_icon(configuration)
        else:
            server = EnfugueServer()
            server.configure(**configuration)
            server.serve()
    finally:
        if open_browser_thread is not None:
            open_browser_thread.stop()
            open_browser_thread.join()
        if server is not None:
            try:
                server.on_destroy()
            except:
                pass
