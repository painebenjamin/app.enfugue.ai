import os
import re
import requests
import datetime

from typing import TypedDict, List, Dict, Any, Iterator, Optional, Union, cast

from semantic_version import Version
from pibble.api.configuration import APIConfiguration
from pibble.util.files import load_yaml, load_json

__all__ = [
    "VersionDict",
    "check_make_directory",
    "get_local_installation_directory",
    "get_local_config_directory",
    "get_local_static_directory",
    "get_local_configuration",
    "get_version",
    "get_versions",
    "get_pending_versions",
    "find_file_in_directory",
    "find_files_in_directory"
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

def check_make_directory(directory: str) -> None:
    """
    Checks if a directory doesn't exist, and makes it.
    Attempts to be thread-safe.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return
        except Exception as ex:
            if not os.path.exists(directory):
                raise IOError(f"Couldn't create directory `{directory}`: {type(ex).__name__}({ex})")
            return

def get_local_configuration(as_api_configuration: bool = False) -> Union[Dict[str, Any], APIConfiguration]:
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
    if as_api_configuration:
        return APIConfiguration(**configuration)
    return configuration


def parse_version(version_string: str) -> Version:
    """
    Parses a version string to a semantic version.
    Does not choke on post-releases, unlike base semver.
    """
    return Version(".".join(version_string.split(".")[:3]))

def get_version() -> Version:
    """
    Gets the version of enfugue installed.
    """
    import logging

    logger = logging.getLogger("enfugue")
    try:
        local_install = get_local_installation_directory()
        version_file = os.path.join(local_install, "version.txt")
        if os.path.exists(version_file):
            with open(version_file, "r") as fp:
                return parse_version(fp.read())
    except:
        pass

    from importlib.metadata import version, PackageNotFoundError

    try:
        return parse_version(version("enfugue"))
    except PackageNotFoundError:
        return Version("0.0.0")


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

def find_file_in_directory(directory: str, file: str, extensions: Optional[List[str]] = None) -> Optional[str]:
    """
    Finds a file in a directory and returns it.
    Uses breadth-first search.
    """
    if not os.path.isdir(directory):
        return None
    if extensions is None:
        file, current_ext = os.path.splitext(file)
        extensions = [current_ext]
    for ext in extensions:
        check_file = os.path.join(directory, f"{file}{ext}")
        if os.path.exists(check_file):
            return check_file
    for filename in os.listdir(directory):
        check_path = os.path.join(directory, filename)
        if os.path.isdir(check_path):
            check_recursed = find_file_in_directory(check_path, file, extensions=extensions)
            if check_recursed is not None:
                return os.path.abspath(check_recursed)
    return None

def find_files_in_directory(directory: str, pattern: Optional[Union[str, re.Pattern]] = None) -> Iterator[str]:
    """
    Find files in a directory, optionally matching a pattern.
    """
    if pattern is not None and isinstance(pattern, str):
        pattern = re.compile(pattern)
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            check_path = os.path.join(directory, filename)
            if os.path.isdir(check_path):
                for sub_file in find_files_in_directory(check_path, pattern):
                    yield sub_file
            elif pattern is None or bool(pattern.match(filename)):
                yield os.path.abspath(check_path)
