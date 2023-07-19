import os
import requests

from enfugue.util.log import logger

__all__ = ["check_download", "check_download_to_dir"]


def check_download(remote_url: str, local_path: str, chunk_size: int = 8192, check_size: bool = True) -> None:
    """
    Checks if a file exists.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.
    """
    if os.path.exists(local_path) and check_size:
        expected_length = requests.head(remote_url, allow_redirects=True).headers.get("Content-Length", None)
        actual_length = os.path.getsize(local_path)
        if expected_length and actual_length != int(expected_length):
            logger.info(
                f"File at {local_path} looks like an interrupted download, or the remote resource has changed. Removing."
            )
            os.remove(local_path)

    if not os.path.exists(local_path):
        logger.info(f"Downloading file from {remote_url}. Will write to {local_path}")
        response = requests.get(remote_url, allow_redirects=True, stream=True)
        with open(local_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                fh.write(chunk)


def check_download_to_dir(remote_url: str, local_dir: str, chunk_size: int = 8192, check_size: bool = True) -> str:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.
    """
    file_name = os.path.basename(remote_url)
    local_path = os.path.join(local_dir, file_name)
    check_download(remote_url, local_path, chunk_size=chunk_size, check_size=check_size)
    return local_path
