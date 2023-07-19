from typing import Any

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
    "CheckResolveURLHelperFilter",
]


def check_url(configuration: APIConfiguration, value: Any) -> Any:
    """
    For configuration values that start with '/', prepend the CMS root.
    For configuration values that start with '/static', prepend the static root.
    """
    if isinstance(value, str) and value.startswith("/"):
        if value.startswith("/static/"):
            static_path = str(configuration.get("server.cms.path.static", configuration["server.cms.path.root"]))
            if static_path.endswith("static") or static_path.endswith("static/"):
                value = value[len("/static") :]
            return url_join(static_path, value)
        else:
            return url_join(str(configuration["server.cms.path.root"]), decode(value))
    return value


class HTMLPropertiesHelperFunction(FunctionExtensionBase):
    """
    Allows the transformation of a dictionary of properties into HTML key=value syntax.

    Callable in templates via {{ html_properties({"foo": "bar"}) }}

    >>> from enfugue.interface.helpers import HTMLPropertiesHelperFunction
    >>> from pibble.api.configuration import APIConfiguration
    >>> config = APIConfiguration(server = {'cms':{'path':{'root':'http://www.example.com'}}})
    >>> self = type('', (object,), {'getConfiguration': lambda: config})
    >>> HTMLPropertiesHelperFunction.__call__(self, {"id": "something", "name": "somethingelse"})
    'id="something" name="somethingelse"'
    >>> HTMLPropertiesHelperFunction.__call__(self, {"src": "/static/img.jpg"})
    'src="http://www.example.com/static/img.jpg"'
    """

    name = "html_properties"

    def __call__(self, property_dict: dict) -> str:
        for property_name in property_dict:
            property_dict[property_name] = check_url(self.getConfiguration(), property_dict[property_name])

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

    >>> from enfugue.interface.helpers import SerializeHelperFunction
    >>> SerializeHelperFunction.__call__(None, {"id": "something", "name": "somethingelse"})
    '{"id": "something", "name": "somethingelse"}'
    >>> SerializeHelperFunction.__call__(None, {"src": None, "length": 4})
    '{"src": null, "length": 4}'
    """

    name = "serialize"

    def __call__(self, var: Any, **kwargs: Any):
        return Serializer.serialize(var, **kwargs)


class CheckResolveURLHelperFilter(FilterExtensionBase):
    """
    Allows calling check_resolve_url() in the template layer.

    Example usage is {{ img | check_resolve_url }}.

    >>> from enfugue.interface.helpers import CheckResolveURLHelperFilter
    >>> from pibble.api.configuration import APIConfiguration
    >>> config = APIConfiguration(server = {'cms':{'path':{'root':'http://www.example.com'}}})
    >>> self = type('', (object,), {'getConfiguration': lambda: config})
    >>> CheckResolveURLHelperFilter.__call__(self, "something")
    'something'
    >>> CheckResolveURLHelperFilter.__call__(self, 4)
    4
    >>> CheckResolveURLHelperFilter.__call__(self, "/static/img.jpg")
    'http://www.example.com/static/img.jpg'
    """

    name = "check_resolve_url"

    def __call__(self, var):
        return check_url(self.getConfiguration(), var)


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
    "CheckResolveURLHelperFilter",
    "SerializeHelperFunction",
    "SerializeHelperFilter",
]
