from typing import Any, Dict

from copy import deepcopy

from pibble.util.strings import Serializer, decode
from pibble.util.helpers import url_join
from pibble.api.configuration import APIConfiguration
from pibble.api.server.webservice.template.extensions import (
    FunctionExtensionBase,
    FilterExtensionBase,
)

__all__ = [
    "HTMLPropertiesHelperFunction",
    "SerializeHelperFunction",
    "SerializeHelperFilter",
    "CheckResolveURLHelperFunction",
]


def check_url(configuration: APIConfiguration, value: Any, paths: Dict[str, str]) -> Any:
    """
    For configuration values that start with '/', prepend the CMS root.
    For configuration values that start with '/static', prepend the static root.
    """
    if isinstance(value, str) and value.startswith("/"):
        if value.startswith("/static/"):
            static_path = paths.get("static", paths["root"])
            if static_path.endswith("static") or static_path.endswith("static/"):
                value = value[len("/static") :]
            return url_join(static_path, value)
        else:
            return url_join(str(paths["root"]), decode(value))
    return value


class HTMLPropertiesHelperFunction(FunctionExtensionBase):
    """
    Allows the transformation of a dictionary of properties into HTML key=value syntax.

    Callable in templates via {{ html_properties({"foo": "bar"}) }}
    """

    name = "html_properties"

    def __call__(self, property_dict: dict, paths: dict) -> str:
        property_dict = deepcopy(property_dict)
        for property_name in property_dict:
            property_dict[property_name] = check_url(self.getConfiguration(), property_dict[property_name], paths)

        return " ".join(
            [
                '{0}="{1}"'.format(key, property_dict[key]) if property_dict[key] is not None else key
                for key in property_dict
            ]
        )


class SerializeHelperFunction(FunctionExtensionBase):
    """
    Allows calling serialize in templates.

    Example usage is {{ serialize({"foo: "bar"}) }}.
    """

    name = "serialize"

    def __call__(self, var: Any, **kwargs: Any):
        return Serializer.serialize(var, **kwargs)


class CheckResolveURLHelperFunction(FunctionExtensionBase):
    """
    Allows calling check_resolve_url() in the template layer.

    Example usage is {{ check_resolve_url(img, paths) }}.
    """

    name = "check_resolve_url"

    def __call__(self, var, paths):
        return check_url(self.getConfiguration(), var, paths)


class SerializeHelperFilter(FilterExtensionBase):
    """
    Allows using serialize as a filter in templates.

    Example usage is {{ {"foo: "bar"} | serialize }}.
    """

    name = "serialize"

    def __call__(self, var: Any):
        return Serializer.serialize(var)


__all__ = [
    "HTMLPropertiesHelperFunction",
    "CheckResolveURLHelperFunction",
    "SerializeHelperFunction",
    "SerializeHelperFilter",
]
