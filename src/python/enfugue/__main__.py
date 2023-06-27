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


@main.command(short_help="Runs the server.")
def run() -> None:
    """
    Runs the server synchronously using cherrypy.
    """
    from enfugue.util import get_local_configuration
    from enfugue.server import EnfugueServer

    configuration = get_local_configuration()
    server = EnfugueServer()
    server.configure(**configuration)
    server.serve()


try:
    main()
except Exception as ex:
    print(termcolor.colored(str(ex), "red"))
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print(termcolor.colored(traceback.format_exc(), "red"))
    sys.exit(5)
sys.exit(0)
