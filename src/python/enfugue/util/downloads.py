import os
import requests

from typing import Optional, Callable

from enfugue.util.log import logger

__all__ = ["check_download", "check_download_to_dir"]

def check_download(
    remote_url: str,
    local_path: str,
    chunk_size: int=8192,
    check_size: bool=True,
    progress_callback: Optional[Callable[[int, int], None]]=None,
) -> None:
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
                f"File at {local_path} looks like an interrupted download, or the remote resource has changed - expected a size of {expected_length} bytes but got {actual_length} instead. Removing."
            )
            os.remove(local_path)

    if not os.path.exists(local_path):
        logger.info(f"Downloading file from {remote_url}. Will write to {local_path}")
        response = requests.get(remote_url, allow_redirects=True, stream=True)
        content_length: Optional[int] = response.headers.get("Content-Length", None) # type: ignore[assignment]
        if content_length is not None:
            content_length = int(content_length)
        with open(local_path, "wb") as fh:
            written_bytes = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                if progress_callback is not None and content_length is not None:
                    written_bytes = min(written_bytes + chunk_size, content_length)
                    progress_callback(written_bytes, content_length)

def check_download_to_dir(
    remote_url: str,
    local_dir: str,
    file_name: Optional[str]=None,
    chunk_size: int=8192,
    check_size: bool=True,
    progress_callback: Optional[Callable[[int, int], None]]=None,
) -> str:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.
    """
    if file_name is None:
        file_name = os.path.basename(remote_url)
    local_path = os.path.join(local_dir, file_name)
    check_download(
        remote_url,
        local_path,
        chunk_size=chunk_size,
        check_size=check_size,
        progress_callback=progress_callback
    )
    return local_path
