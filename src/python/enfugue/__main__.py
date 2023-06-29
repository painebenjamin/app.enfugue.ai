import os
import sys
import tqdm
import click
import termcolor
import logging
import traceback

from typing import Optional
from PIL import Image
from pibble.util.strings import Serializer
from pibble.util.log import LevelUnifiedLoggingContext
from pibble.api.configuration import APIConfiguration
from enfugue.util import logger


TRT_PIPELINE = "enfugue.diffusion.rt.pipeline.EnfugueTensorRTStableDiffusionPipeline"


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

    click.echo(f"Enfugue v.{enfugue_version}")
    click.echo(f"Torch v.{torch.__version__}")
    click.echo("CUDA {0}".format("available" if torch.cuda.is_available() else "unavailable"))
    try:
        import tensorrt

        click.echo("TensorRT Supported")
    except ImportError:
        click.echo("TensorRT Unsupported")

@main.command(short_help = "Dumps a copy of the configuration")
@click.option("-f", "--filename", help="A file to write to instead of stdout.")
@click.option("-j", "--json", help="When passed, use JSON instead of YAML.", is_flag=True, default=False)
def dump_config(
    filename: Optional[str] = None,
    json: bool = False
) -> None:
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
        print(f"Wrote {filename}")
        return
    if json:
        import json
        print(json.dumps(configuration, indent=4))
    else:
        import yaml
        print(yaml.dump(configuration))


@click.option("-c", "--config", help="An optional path to a configuration file to use instead of the default.")
@main.command(short_help="Runs the server.")
def run(config: str = None) -> None:
    """
    Runs the server synchronously using cherrypy.
    """
    from enfugue.server import EnfugueServer
    
    if config is not None:
        if config.endswith("json"):
            from pibble.util.files import load_json
            configuration = json_file(config)
        elif config.endswith("yml") or config.endswith("yaml"):
            from pibble.util.files import load_yaml
            configuration = load_yaml(config)
        else:
            raise IOError(f"Unknown format for configuration file {config}")
    else:
        from enfugue.util import get_local_configuration
        configuration = get_local_configuration()

    server = EnfugueServer()
    server.configure(**configuration)
    
    secure = configuration["server"].get("secure", False)
    domain = configuration["server"].get("domain", "127.0.0.1")
    port = configuration["server"].get("port", 45554)
    
    scheme = "https" if secure else "http"
    port_print = "" if port in [80, 443] else f":{port}"
    print(f"Running enfugue, visit at {scheme}://{domain}{port_print}/ (press Ctrl+C to exit)")
    server.serve()
    print("Goodbye!")

try:
    main()
except Exception as ex:
    print(termcolor.colored(str(ex), "red"))
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print(termcolor.colored(traceback.format_exc(), "red"))
    sys.exit(5)
sys.exit(0)
