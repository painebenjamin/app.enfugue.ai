import os
import sys
import json
import click
import termcolor
import logging
import traceback

from typing import Optional
from PIL import Image
from pibble.util.strings import Serializer
from pibble.util.log import LevelUnifiedLoggingContext
from pibble.util.files import load_json, load_yaml
from pibble.api.configuration import APIConfiguration

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
@click.option("-o", "--overrides", help="An optional JSON object containing override configuration.")
@main.command(short_help="Runs the server.")
def run(
    config: str = None,
    merge: bool = False,
    overrides: str = None
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

    if overrides:
        overrides = json.loads(overrides)
        merge_into(overrides, configuration)

    server = EnfugueServer()
    server.configure(**configuration)

    secure = configuration["server"].get("secure", False)
    domain = configuration["server"].get("domain", "127.0.0.1")
    port = configuration["server"].get("port", 45554)

    scheme = "https" if secure else "http"
    port_echo = "" if port in [80, 443] else f":{port}"
    click.echo(f"Running enfugue, visit at {scheme}://{domain}{port_echo}/ (press Ctrl+C to exit)")
    server.serve()
    click.echo("Goodbye!")


try:
    main()
except Exception as ex:
    print(termcolor.colored(str(ex), "red"))
    if "--verbose" in sys.argv or "-v" in sys.argv or os.environ.get("ENFUGUE_DEBUG", "0") == "1":
        print(termcolor.colored(traceback.format_exc(), "red"))
    sys.exit(5)
sys.exit(0)
