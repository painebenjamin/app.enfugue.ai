import os
import sys
import time
import json
import click
import termcolor
import logging
import traceback

from typing import Optional, Any, List

from PIL import Image
from copy import deepcopy

from pibble.util.strings import Serializer
from pibble.util.log import LevelUnifiedLoggingContext
from pibble.util.files import load_json, load_yaml
from pibble.api.configuration import APIConfiguration
from pibble.api.exceptions import ConfigurationError

from enfugue.util import logger, merge_into, get_local_configuration

@click.group(name="enfugue")
def main() -> None:
    """
    Enfugue command-line tools.
    """
    pass

@main.command(short_help="Prints the version of the main enfugue package and dependant packages.")
def version() -> None:
    """
    Gets the version of packages installed in the environment.
    """
    from enfugue.util import get_version

    try:
        enfugue_version = get_version()
    except:
        enfugue_version = "development"

    import torch
    from enfugue.diffusion.util import (
        get_optimal_device,
        tensorrt_available,
        directml_available,
    )

    device = get_optimal_device()

    click.echo(f"Enfugue v.{enfugue_version}")
    click.echo(f"Torch v.{torch.__version__}")
    click.echo(f"\nAI/ML Capabilities:\n---------------------")
    click.echo(f"Device type: {device.type}")

    if torch.cuda.is_available():
        if torch.backends.cuda.is_built():
            click.echo(f"CUDA: v.{torch.version.cuda} Ready")
        else:
            click.echo("CUDA: Available, but not installed")
        if tensorrt_available():
            import tensorrt
            click.echo(f"TensorRT: v.{tensorrt.__version__} Ready")
        else:
            click.echo("TensorRT: Unavailable")
    else:
        click.echo("CUDA: Unavailable")

    if torch.backends.cudnn.is_available():
        click.echo(f"CUDNN: v.{torch.backends.cudnn.version()} Ready")
    else:
        click.echo("CUDNN: Unavailable")

    if directml_available():
        click.echo("DirectML: Ready")
    else:
        click.echo("DirectML: Unavailable")

    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            click.echo("MPS: Ready")
        else:
            click.echo("MPS: Available, but not installed")
    else:
        click.echo("MPS: Unavailable")


@main.command(short_help="Dumps a copy of the configuration")
@click.option("-f", "--filename", help="A file to write to instead of stdout.")
@click.option("-j", "--json", help="When passed, use JSON instead of YAML.", is_flag=True, default=False)
def dump_config(filename: Optional[str] = None, json: bool = False) -> None:
    """
    Dumps a copy of the configuration to the console or the specified path.
    """
    from enfugue.util import get_local_configuration

    configuration = get_local_configuration()
    if filename is not None:
        if json:
            from pibble.util.files import dump_json

            dump_json(filename, configuration)
        else:
            from pibble.util.files import dump_yaml

            dump_yaml(filename, configuration)
        click.echo(f"Wrote {filename}")
        return
    if json:
        import json

        click.echo(json.dumps(configuration, indent=4))
    else:
        import yaml

        click.echo(yaml.dump(configuration))


@click.option("-c", "--config", help="An optional path to a configuration file to use instead of the default.")
@click.option("-m", "--merge", is_flag=True, default=False, help="When set, merge the passed configuration with the default configuration instead of replacing it.")
@click.option("-o", "--overrides", help="an optional json object containing override configuration.")
@click.option("-d", "--debug", help="Enable debug logging.", is_flag=True, default=False)
@main.command(short_help="Runs the server.")
def run(
    config: str = None,
    merge: bool = False,
    overrides: str = None,
    debug: bool = False
) -> None:
    """
    Runs the server synchronously using cherrypy.
    """
    from enfugue.server import EnfugueServer

    if config is not None:
        if config.endswith("json"):

            configuration = json_file(config)
        elif config.endswith("yml") or config.endswith("yaml"):
            from pibble.util.files import load_yaml

            configuration = load_yaml(config)
        else:
            raise IOError(f"Unknown format for configuration file {config}")

        if merge:
            configuration = merge_into(configuration, get_local_configuration())
    else:
        configuration = get_local_configuration()

    while "configuration" in configuration:
        configuration = configuration["configuration"]

    if overrides:
        overrides = json.loads(overrides)
        merge_into(overrides, configuration)
    if debug:
        log_overrides = {
            "logging": {
                "level": "debug",
                "handler": "stream",
                "stream": "stdout",
                "colored": True
             }
        }
        merge_into(log_overrides, configuration)

    # Determine how many servers we need to run
    try:

        all_host = configuration["server"]["host"]
        all_port = configuration["server"]["port"]
        all_domain = configuration["server"].get("domain", None)
        all_secure = configuration["server"].get("secure", False)
        all_cert = configuration["server"].get("cert", None)
        all_key = configuration["server"].get("key", None)
    except KeyError as ex:
        if debug:
            click.echo(termcolor.colored(json.dumps(configuration, indent=4), "cyan"))
        raise ConfigurationError(f"Missing required configuration key {ex}")

    num_servers = 1
    for var in [all_host, all_secure, all_domain, all_port, all_cert, all_key]:
        if isinstance(var, list) or isinstance(var, tuple):
            num_servers = max(num_servers, len(var))

    def get_for_index(index: int, config_value: Any) -> Any:
        if isinstance(config_value, list) or isinstance(config_value, tuple):
            if index < len(config_value):
                return config_value[index]
            return config_value[-1]
        return config_value

    servers: List[EnfugueServer] = []

    try:
        for i in range(num_servers):
            host = get_for_index(i, all_host)
            port = get_for_index(i, all_port)
            secure = get_for_index(i, all_secure)
            cert = get_for_index(i, all_cert)
            key = get_for_index(i, all_key)
            domain = get_for_index(i, all_domain)

            this_configuration = deepcopy(configuration)
            this_configuration["server"]["host"] = host
            this_configuration["server"]["port"] = port
            this_configuration["server"]["domain"] = domain
            this_configuration["server"]["secure"] = secure
            this_configuration["server"]["cert"] = cert
            this_configuration["server"]["key"] = key

            server = EnfugueServer()

            scheme = "https" if secure else "http"
            port_echo = "" if port in [80, 443] else f":{port}"
            domain_echo = host if not domain else domain
            if num_servers == 1:
                server.configure(**this_configuration)
                click.echo(f"Running enfugue, visit at {scheme}://{domain_echo}{port_echo}/ (press Ctrl+C to exit)")
                server.serve()
                return
            click.echo(f"Server {(i+1):d} of {num_servers:d} listening on {scheme}://{domain_echo}{port_echo}/ (press Ctrl+C to exit)")
            server.configure_start(**this_configuration)
            servers.append(server)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        click.echo(termcolor.colored(str(ex), "red"))
        if debug:
            click.echo(termcolor.colored(traceback.format_exc(), "red"))
    finally:
        for server in servers:
            try:
                server.stop()
            except:
                pass
        click.echo("Goodbye!")

try:
    main()
    sys.exit(0)
except Exception as ex:
    sys.exit(5)
