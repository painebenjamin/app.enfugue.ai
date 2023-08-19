import os
import time
import datetime
from typing import Optional, Dict, Callable, Tuple, Any, cast
from multiprocessing import Process, Queue as MakeQueue
from multiprocessing.queues import Queue
from queue import Empty
from enfugue.util import logger
from pibble.util.strings import get_uuid

__all__ = ["DownloadProcess", "Download"]


class DownloadProcess(Process):
    """
    A process to download a file to a location, optionally reporting progress.
    """

    def __init__(
        self,
        src: str,
        dest: str,
        chunk_size: int = 8192,
        headers: Dict[str, str] = {},
        parameters: Dict[str, Any] = {},
        progress: Optional[Queue] = None,
    ) -> None:
        super(DownloadProcess, self).__init__()
        self.src = src
        self.dest = dest
        self.chunk_size = chunk_size
        self.headers = headers
        self.parameters = parameters
        self.progress = progress

    def run(self) -> None:
        """
        Runs the download process, optionally sending progress over the queue.
        """
        from requests import get

        try:
            start = datetime.datetime.now()
            elapsed: Callable[[], float] = lambda: (datetime.datetime.now() - start).total_seconds()
            response = get(self.src, headers=self.headers, params=self.parameters, stream=True)
            try:
                length = int(response.headers["Content-Length"])
            except KeyError:
                logger.warning(f"No content-length found in download. Exiting. Headers were: {response.headers}")
                raise IOError(f"URL {self.src} did not respond with a content-length, cannot download.")
            if self.progress is not None:
                self.progress.put_nowait((elapsed(), 0, length))
            with open(self.dest, "wb") as fh:
                for i, chunk in enumerate(response.iter_content(chunk_size=self.chunk_size)):
                    fh.write(chunk)
                    if self.progress is not None:
                        self.progress.put_nowait((elapsed(), (i + 1) * self.chunk_size, length))
            if self.progress is not None:
                self.progress.put_nowait((elapsed(), length, length))
        except Exception as ex:
            logger.critical(f"Received critical error during download: {type(ex).__name__} {ex}")
            raise


class Download:
    """
    This class holds the queue and the process for downloading a file.
    """

    progress_queue: Optional[Queue]
    create_time: datetime.datetime
    start_time: Optional[datetime.datetime]
    close_time: Optional[datetime.datetime]
    last_elapsed: Optional[float]
    last_downloaded_bytes: Optional[int]
    last_total_bytes: Optional[int]

    def __init__(
        self,
        src: str,
        dest: str,
        chunk_size: int = 8192,
        headers: Dict[str, str] = {},
        parameters: Dict[str, Any] = {},
        progress: bool = True,
    ) -> None:
        self.id = get_uuid()
        self.src = src
        self.dest = dest
        self.create_time = datetime.datetime.now()
        self.progress_queue = None if not progress else MakeQueue()
        self.start_time = None
        self.close_time = None
        self.last_elapsed = None
        self.last_downloaded_bytes = None
        self.last_total_bytes = None
        self.process = DownloadProcess(src, dest, chunk_size, headers, parameters, self.progress_queue)
        self.canceled = False

    def start(self) -> None:
        """
        Starts the process, and sets the start time.
        """
        if self.start_time is not None:
            raise IOError("Process already started.")
        logger.debug(f"Starting download process for {self.src}")
        self.start_time = datetime.datetime.now()
        self.process.start()

    def close(self) -> None:
        """
        Closes the queue and sets close time.
        """
        if self.closed:
            raise IOError("Already closed!")
        self.close_time = datetime.datetime.now()
        if self.progress_queue:
            self.progress_queue.close()
        if self.process.is_alive():
            logger.warning("Closed while process was still alive. Trying to terminate.")
            self.process.terminate()

    def get_last_progress(
        self,
    ) -> Optional[Tuple[Optional[float], Optional[int], Optional[int]]]:
        """
        Checks if there is new progress data from the queue, reads it all, then returns the last.
        """
        if not self.started:
            return None
        if self.progress_queue is None:
            raise ValueError("Download was initialized with `progress=False`, no progress report is available.")
        if not self.closed:
            last_tuple = None
            try:
                while True:
                    last_tuple = self.progress_queue.get_nowait()
            except Empty:
                if not self.process.is_alive():
                    self.close()
            if last_tuple is not None:
                (
                    self.last_elapsed,
                    self.last_downloaded_bytes,
                    self.last_total_bytes,
                ) = last_tuple
        return self.last_elapsed, self.last_downloaded_bytes, self.last_total_bytes

    def check_raise_exitcode(self) -> None:
        """
        Checks if the process is alive, and if not, checks its exit code.
        If the process exited with an error, this will an IOError.
        """
        if self.started and not self.process.is_alive() and self.process.exitcode != 0:
            raise IOError(f"Process died with exit code {self.process.exitcode}")

    def format(self) -> Dict[str, Any]:
        """
        Gets the entire details of the download in one dictionary.
        """
        elapsed, downloaded, total = None, None, None
        status = "canceled" if self.canceled else "queued"
        if self.started and not self.canceled:
            if self.closed:
                status = "complete"
            elif self.started and not self.process.is_alive():
                status = "error"
            elif self.process.is_alive():
                status = "downloading"
            else:
                status = "unknown"
            last_progress = self.get_last_progress()
            if last_progress is not None:
                elapsed, downloaded, total = last_progress
        return {
            "id": self.id,
            "status": status,
            "source": self.src,
            "destination": self.dest,
            "filename": os.path.basename(self.dest),
            "elapsed": elapsed,
            "downloaded": downloaded,
            "total": total,
        }

    def cancel(self) -> None:
        """
        Cancels this download.
        """
        if not self.complete:
            self.close()
            self.canceled = True
            remove_attempts = 0
            max_remove_attempts = 10
            while os.path.exists(self.dest):
                try:
                    os.unlink(self.dest)
                except Exception as ex:
                    remove_attempts += 1
                    if remove_attempts >= max_remove_attempts:
                        logger.error(f"Couldn't remove {self.dest} after {max_remove_attempts} tries. Last error was {ex}")
                        return
                    time.sleep(0.2)

    @property
    def started(self) -> bool:
        """
        A quick helper for a boolean 'started' attribute
        """
        return self.start_time is not None

    @property
    def closed(self) -> bool:
        """
        Returns true if close time is set and queue is closed.
        """
        return self.close_time is not None

    @property
    def complete(self) -> bool:
        """
        Determines if the download is completed.
        """
        if not self.started:
            return False
        if self.closed:
            return True
        self.check_raise_exitcode()
        self.get_last_progress()
        return not self.process.is_alive()

    @property
    def progress(self) -> Optional[float]:
        """
        Gets the last progress from 0 to 1.0.
        Raises errors if the process failed.
        """
        if not self.started:
            return None
        self.check_raise_exitcode()
        self.get_last_progress()
        if self.last_downloaded_bytes is None or self.last_total_bytes is None:
            return None
        return self.last_downloaded_bytes / self.last_total_bytes

    @property
    def downloaded_bytes(self) -> Optional[int]:
        """
        Gets the last number of bytes downloaded.
        Raises errors if the process failed.
        """
        if not self.started:
            return None
        self.check_raise_exitcode()
        self.get_last_progress()
        return self.last_downloaded_bytes

    @property
    def total_bytes(self) -> Optional[int]:
        """
        Gets the last total number of bytes.
        Raises errors if the process failed.
        """
        if not self.started:
            return None
        self.check_raise_exitcode()
        self.get_last_progress()
        return self.last_total_bytes

    @property
    def elapsed(self) -> Optional[float]:
        """
        Gets the last total number of bytes.
        Raises errors if the process failed.
        """
        if not self.started:
            return None
        self.check_raise_exitcode()
        self.get_last_progress()
        if not self.last_elapsed:
            return (datetime.datetime.now() - cast(datetime.datetime, self.start_time)).total_seconds()
        return self.last_elapsed
