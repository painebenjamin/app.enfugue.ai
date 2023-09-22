from typing import Dict, Any

__all__ = [
    "noop",
    "replace_images",
    "redact_for_log"
]

def noop(*args: Any, **kwargs: Any) -> None:
    """
    Does nothing.
    """

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

def redact_for_log(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redacts prompts from logs to encourage log sharing for troubleshooting.
    """
    redacted = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            redacted[key] = redact_for_log(value)
        elif isinstance(value, tuple):
            redacted[key] = "(" + ", ".join([str(redact_for_log({"v": v})["v"]) for v in value]) + ")" # type: ignore[assignment]
        elif isinstance(value, list):
            redacted[key] = "[" + ", ".join([str(redact_for_log({"v": v})["v"]) for v in value]) + "]" # type: ignore[assignment]
        elif type(value) not in [str, float, int, bool, type(None)]:
            redacted[key] = type(value).__name__ # type: ignore[assignment]
        elif "prompt" in key and value is not None:
            redacted[key] = "***" # type: ignore[assignment]
        else:
            redacted[key] = str(value) # type: ignore[assignment]
    return redacted
