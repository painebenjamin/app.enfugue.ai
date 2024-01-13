import math

from typing import Dict, Any, Iterator, List, Iterable, Optional
from contextlib import contextmanager

__all__ = [
    "noop",
    "merge_into",
    "replace_images",
    "redact_for_log",
    "profiler",
    "reiterator",
    "human_duration"
]

def noop(*args: Any, **kwargs: Any) -> None:
    """
    Does nothing.
    """

def merge_into(source: Dict[str, Any], dest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges a source dictionary into a target dictionary.

    >>> from enfugue.util.misc import merge_into
    >>> x = {"a": 1}
    >>> merge_into({"b": 2}, x)
    {'a': 1, 'b': 2}
    """
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(dest.get(key, None), dict):
            merge_into(source[key], dest[key])
        else:
            dest[key] = source[key]
    return dest

def replace_images(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replaces images in a dictionary with a metadata dictionary.
    """
    from PIL.Image import Image
    for key, value in dictionary.items():
        if isinstance(value, Image):
            width, height = value.size
            metadata = {"width": width, "height": height, "mode": value.mode}
            if hasattr(value, "filename"):
                metadata["filename"] = value.filename
            if hasattr(value, "text"):
                metadata["text"] = value.text
            dictionary[key] = metadata
        elif isinstance(value, dict):
            dictionary[key] = replace_images(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            dictionary[key] = [
                replace_images(part) if isinstance(part, dict) else part
                for part in value
            ]
            if isinstance(value, tuple):
                dictionary[key] = tuple(dictionary[key])
    return dictionary

def redact_for_log(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redacts prompts from logs to encourage log sharing for troubleshooting.
    """
    from PIL.Image import Image
    redacted: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "audio":
            frequency_bands = len(value[0])
            audio_samples = len(value[1])
            channels = len(value[1][0][0])
            redacted[key] = f"Audio({audio_samples} samples, {frequency_bands} frequency bands, {channels} channel(s))"
        elif key == "prompts" and value:
            total = len(value)
            redacted[key] = f"PromptList({total} prompt(s))"
        elif key == "motion_vectors" and value:
            total = len(value)
            redacted[key] = f"MotionVectors({total} vector(s))"
        elif isinstance(value, dict):
            redacted[key] = redact_for_log(value)
        elif isinstance(value, tuple):
            redacted[key] = "(" + ", ".join([str(redact_for_log({"v": v})["v"]) for v in value]) + ")" # type: ignore[assignment]
        elif isinstance(value, list):
            if value and isinstance(value[0], Image):
                total = len(value)
                redacted[key] = f"ImageList({total} image(s))"
            else:
                redacted[key] = "[" + ", ".join([str(redact_for_log({"v": v})["v"]) for v in value]) + "]" # type: ignore[assignment]
        elif type(value) not in [str, float, int, bool, type(None)]:
            redacted[key] = type(value).__name__ # type: ignore[assignment]
        elif "prompt" in key and "num" not in key and value is not None:
            redacted[key] = "***" # type: ignore[assignment]
        else:
            redacted[key] = str(value) # type: ignore[assignment]
    return redacted

@contextmanager
def profiler() -> Iterator:
    """
    Runs a profiler.
    """
    from cProfile import Profile
    from pstats import SortKey, Stats
    with Profile() as profile:
        yield
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()

class reiterator:
    """
    Transparently memoized any iterator
    """
    memoized: List[Any]

    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable = iterable
        self.memoized = []
        self.started = False
        self.finished = False

    def __iter__(self) -> Iterable[Any]:
        if not self.started:
            self.started = True
            last_index: Optional[int] = None
            for i, value in enumerate(self.iterable):
                yield value
                self.memoized.append(value)
                last_index = i
                if self.finished:
                    # Completed somewhere else
                    break
            if self.finished:
                if last_index is None:
                    last_index = 0
                for value in self.memoized[last_index+1:]:
                    yield value
            self.finished = True
            del self.iterable
        elif not self.finished:
            # Complete iterator
            self.memoized += [item for item in self.iterable]
            self.finished = True
            del self.iterable
            for item in self.memoized:
                yield item
        else:
            for item in self.memoized:
                yield item

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = SECONDS_PER_MINUTE*60
SECONDS_PER_DAY = SECONDS_PER_HOUR*24

def human_duration(seconds: int, trim: bool = True, compact: bool = False) -> str:
    """
    Turns some number of seconds into a string
    """
    days = 0
    hours = 0
    minutes = 0

    if seconds > SECONDS_PER_DAY:
        days = math.floor(seconds / SECONDS_PER_DAY)
        seconds -= days * SECONDS_PER_DAY

    if seconds > SECONDS_PER_HOUR:
        hours = math.floor(seconds / SECONDS_PER_HOUR)
        seconds -= hours * SECONDS_PER_HOUR

    if seconds > SECONDS_PER_MINUTE:
        minutes = math.floor(seconds / SECONDS_PER_MINUTE)
        seconds -= minutes * SECONDS_PER_MINUTE

    seconds = math.floor(seconds)

    if compact:
        duration_separator = ":"
        duration_string = f"{days}:{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        duration_separator = ", "
        days_plural = "s" if days != 1 else ""
        hours_plural = "s" if hours != 1 else ""
        minutes_plural = "s" if minutes != 1 else ""
        seconds_plural = "s" if seconds != 1 else ""
        duration_string = f"{days} day{days_plural}, {hours} hour{hours_plural}, {minutes} minute{minutes_plural}, {seconds} second{seconds_plural}"

    if trim and days == 0:
        duration_parts = duration_string.split(duration_separator)
        if hours == 0 and minutes == 0:
            if compact:
                return f"0:{duration_parts[-1]}"
            return duration_separator.join(duration_parts[3:])
        elif hours == 0:
            return duration_separator.join(duration_parts[2:])
        else:
            return duration_separator.join(duration_parts[1:])
    return duration_string
