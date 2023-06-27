import os
import requests
import datetime

from typing import TypedDict, List, Dict, Any, cast

from semantic_version import Version
from pibble.util.files import load_yaml, load_json

__all__ = [
    "VersionDict",
    "get_local_installation_directory",
    "get_local_config_directory",
    "get_local_static_directory",
    "get_local_configuration",
    "get_version",
    "get_versions",
    "get_pending_versions",
]


class VersionDict(TypedDict):
    """
    The version dictionary.
    """

    version: Version
    release: datetime.date
    description: str


def get_local_installation_directory() -> str:
    """
    Gets where the local installation directory is (i.e. where the package data files are,
    either in ../site-packages/enfugue on a python installation, or ${ROOT}/enfugue in a
    precompiled package
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while not os.path.basename(here) == "enfugue":
        here = os.path.abspath(os.path.join(here, "../"))
        if here == "/":
            raise IOError("Couldn't find installation directory.")
    return here


def get_local_config_directory() -> str:
    """
    Gets where the local configuration directory is.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(here, "config")):
        here = os.path.abspath(os.path.join(here, "../"))
        if here == "/":
            raise IOError("Couldn't find config directory.")
    return os.path.join(here, "config")


def get_local_static_directory() -> str:
    """
    Gets where the local static directory is.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(here, "static")):
        here = os.path.abspath(os.path.join(here, "../"))
        if here == "/":
            raise IOError("Couldn't find static directory.")
    return os.path.join(here, "static")


def get_local_configuration() -> Dict[str, Any]:
    """
    Gets configuration from a file in the environment, or the base config.
    """
    default_config = os.path.join(get_local_config_directory(), "server.yml")
    config_file = os.getenv("ENFUGUE_CONFIG", default_config)
    if not os.path.exists(config_file):
        raise IOError(f"Configuration file {config_file} missing or inaccessible")

    basename, ext = os.path.splitext(os.path.basename(config_file))
    configuration: Dict[str, Any] = {}
    if ext.lower() in [".yml", ".yaml"]:
        configuration = load_yaml(config_file)
    elif ext.lower() == ".json":
        configuration = load_json(config_file)
    else:
        raise IOError(f"Unknown extension {ext}")
    if "configuration" in configuration:
        configuration = configuration["configuration"]
    return configuration


def get_version() -> Version:
    """
    Gets the version of enfugue installed.
    """
    from importlib.metadata import version, PackageNotFoundError

    try:
        return Version(version("enfugue"))
    except PackageNotFoundError:
        return "development"


def get_versions() -> List[VersionDict]:
    """
    Gets all version details from the CDN.
    """
    version_data = requests.get("https://cdn.enfugue.ai/versions.json").json()
    versions: List[VersionDict] = [
        cast(
            VersionDict,
            {
                "version": Version(datum["version"]),
                "release": datetime.datetime.strptime(datum["release"], "%Y-%m-%d").date(),
                "description": datum["description"],
            },
        )
        for datum in version_data
    ]
    versions.sort(key=lambda v: v["version"])
    return versions


def get_pending_versions() -> List[VersionDict]:
    """
    Gets only versions yet to be installed.
    """
    current_version = get_version()
    return [version for version in get_versions() if version["version"] > current_version]
