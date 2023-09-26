from typing import Dict, Any

__all__ = [
    "merge_into"
]

def merge_into(source: Dict[str, Any], dest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges a source dictionary into a target dictionary.

    >>> from enfugue.util.misc import merge_into
    >>> x = {"a": 1}
    >>> merge_into({"b": 2}, x)
    >>> x["b"]
    2
    """
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(dest.get(key, None), dict):
            merge(source[key], dest[key])
        else:
            dest[key] = source[key]
    return dest
