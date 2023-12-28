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

    # Compatibility fixes
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from enfugue.util.runner import EnfugueServerRunner

    system = platform.system()
    configuration = get_local_configuration()
    runner = EnfugueServerRunner(configuration)

    open_browser: bool = configuration.get("open", os.getenv("ENFUGUE_OPEN", True))
    if isinstance(open_browser, str): # type: ignore
        open_browser = open_browser[0].lower() in ["1", "t", "y"] # type: ignore[unreachable]

    if system == "Windows":
        runner.run_icon(open_browser=open_browser)
    else:
        runner.run(open_browser=open_browser)
