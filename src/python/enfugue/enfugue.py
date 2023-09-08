"""
This is the entry point for the windows .exe
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
    if configuration.get("open", True):
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
