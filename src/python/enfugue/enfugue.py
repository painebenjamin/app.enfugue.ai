"""
This is the entry point for the windows .exe
"""
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # pyinstaller fix

    from enfugue.util import get_local_configuration
    import os
    import platform

    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    from enfugue.server import EnfugueServer

    system = platform.system()
    configuration = get_local_configuration()
    if system == "Windows":
        EnfugueServer.serve_icon(configuration)
    else:
        server = EnfugueServer()
        server.configure(**configuration)
        server.serve()
