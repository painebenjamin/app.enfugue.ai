import os
import requests

from typing import Optional, Callable, Union, BinaryIO, Iterator

from contextlib import contextmanager

from enfugue.util.log import logger
from pibble.util.numeric import human_size
from enfugue.util.misc import human_duration

__all__ = [
    "check_download",
    "check_download_to_dir",
    "get_file_name_from_url",
    "get_domain_from_url",
    "get_download_text_callback"
]

def get_domain_from_url(url: str) -> str:
    """
    Gets a domain from a URL.
    """
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    return parsed_url.netloc

def get_file_name_from_url(url: str) -> str:
    """
    Gets a filename from a URL.
    Used to help with default models that don't have the same filename as their URL
    """
    from urllib.parse import urlparse, parse_qs
    parsed_url = urlparse(url)
    parsed_qs = parse_qs(parsed_url.query)
    if "filename" in parsed_qs:
        return parsed_qs["filename"][0]
    elif "response-content-disposition" in parsed_qs:
        disposition_parts = parsed_qs["response-content-disposition"][0].split(";")
        for part in disposition_parts:
            part_data = part.strip("'\" ").split("=")
            if len(part_data) < 2:
                continue
            part_key, part_value = part_data[0], "=".join(part_data[1:])
            if part_key == "filename":
                return part_value.strip("'\" ")
    return os.path.basename(url.split("?")[0])

def check_download(
    remote_url: str,
    target: Union[str, BinaryIO],
    chunk_size: int=8192,
    check_size: bool=True,
    resume_size: int = 0,
    progress_callback: Optional[Callable[[int, int], None]]=None,
    text_callback: Optional[Callable[[str], None]]=None
) -> None:
    """
    Checks if a file exists.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.
    """
    if isinstance(target, str) and os.path.exists(target) and check_size and resume_size <= 0:
        expected_length = requests.head(remote_url, allow_redirects=True).headers.get("Content-Length", None)
        actual_length = os.path.getsize(target)
        if expected_length and actual_length != int(expected_length):
            logger.info(
                f"File at {target} looks like an interrupted download, or the remote resource has changed - expected a size of {expected_length} bytes but got {actual_length} instead. Removing."
            )
            os.remove(target)

    headers = {}
    if resume_size is not None:
        headers["Range"] = f"bytes={resume_size:d}-"

    if text_callback is not None:
        progress_text_callback = get_download_text_callback(remote_url, text_callback)
        original_progress_callback = progress_callback

        def new_progress_callback(written: int, total: int) -> None:
            progress_text_callback(written, total)
            if original_progress_callback is not None:
                original_progress_callback(written, total)

        progress_callback = new_progress_callback

    if not isinstance(target, str) or not os.path.exists(target):
        @contextmanager
        def get_write_handle() -> Iterator[BinaryIO]:
            if isinstance(target, str):
                with open(target, "wb") as handle:
                    yield handle
            else:
                yield target
        logger.info(f"Downloading file from {remote_url}. Will write to {target}")
        response = requests.get(remote_url, allow_redirects=True, stream=True, headers=headers)
        content_length: Optional[int] = response.headers.get("Content-Length", None) # type: ignore[assignment]
        if content_length is not None:
            content_length = int(content_length)
        with get_write_handle() as fh:
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
    text_callback: Optional[Callable[[str], None]]=None
) -> str:
    """
    Checks if a file exists in a directory based on a remote path.
    If it does, checks the size and matches against the remote URL.
    If it doesn't, or the size doesn't match, download it.
    """
    if file_name is None:
        file_name = get_file_name_from_url(remote_url)

    local_path = os.path.join(local_dir, file_name)

    check_download(
        remote_url,
        local_path,
        chunk_size=chunk_size,
        check_size=check_size,
        progress_callback=progress_callback,
        text_callback=text_callback
    )
    return local_path

def get_download_text_callback(
    url: str,
    callback: Callable[[str], None]
) -> Callable[[int, int], None]:
    """
    Gets the callback that applies during downloads.
    """
    from datetime import datetime

    last_callback = datetime.now()
    last_callback_amount: int = 0
    bytes_per_second_history = []
    file_label = "{0} from {1}".format(
        get_file_name_from_url(url),
        get_domain_from_url(url)
    )

    def progress_callback(written_bytes: int, total_bytes: int) -> None:
        nonlocal last_callback
        nonlocal last_callback_amount
        this_callback = datetime.now()
        this_callback_offset = (this_callback-last_callback).total_seconds()
        if this_callback_offset > 1:
            difference = written_bytes - last_callback_amount

            bytes_per_second = difference / this_callback_offset
            bytes_per_second_history.append(bytes_per_second)
            bytes_per_second_average = sum(bytes_per_second_history[-10:]) / len(bytes_per_second_history[-10:])

            estimated_seconds_remaining = (total_bytes - written_bytes) / bytes_per_second_average
            estimated_duration = human_duration(int(estimated_seconds_remaining), compact=True)
            percentage = (written_bytes / total_bytes) * 100.0
            callback(f"Downloading {file_label}: {percentage:0.1f}% ({human_size(written_bytes)}/{human_size(total_bytes)}), {human_size(bytes_per_second)}/s, {estimated_duration} remaining")
            last_callback = this_callback
            last_callback_amount = written_bytes

    return progress_callback
