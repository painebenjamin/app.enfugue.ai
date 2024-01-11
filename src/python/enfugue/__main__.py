import os
import sys
import time
import json
import math
import click
import termcolor
import logging
import traceback

from typing import Optional, Any, List, Union, Dict, Iterator

from contextlib import contextmanager
from PIL import Image
from copy import deepcopy

from pibble.util.strings import Serializer
from pibble.util.log import LevelUnifiedLoggingContext
from pibble.util.files import load_json, load_yaml
from pibble.api.configuration import APIConfiguration
from pibble.api.exceptions import ConfigurationError

from enfugue.util import logger, merge_into, get_local_configuration

@contextmanager
def get_context(debug: bool=False) -> Iterator:
    """
    Either blank or debug context manager
    """
    if debug:
        from pibble.util.log import DebugUnifiedLoggingContext
        with DebugUnifiedLoggingContext():
            yield
    else:
        yield

def get_configuration(
    config: Optional[str]=None,
    overrides: Optional[Union[str,Dict[str, Any]]]=None,
    merge: bool=False,
    debug: bool=False,
) -> Dict[str, Any]:
    """
    Gets the configuration to use based on passed arguments.
    """
    if config is not None:
        if config.endswith("json"):
            configuration = load_json(config)
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
        if isinstance(overrides, str):
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

    return configuration

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
@click.option("-r", "--role", help="An optional role of the conversation to use instead of the default.")
@click.option("-s", "--system", help="An optional system message to send instead of the default for the chosen role (or default.)")
@click.option("-m", "--merge", is_flag=True, default=False, help="When set, merge the passed configuration with the default configuration instead of replacing it.")
@click.option("-d", "--debug", help="Enable debug logging.", is_flag=True, default=False)
@click.option("-u", "--unsafe", help="Disable safety.", is_flag=True, default=False)
@click.option("-t", "--temperature", help="The temperature (randomness) of the responses.", default=0.7, show_default=True)
@click.option("-k", "--top-k", help="The number of tokens to limit to when selecting the next token in a response.", default=50, show_default=True)
@click.option("-p", "--top-p", help="The p-value of tokens to limit from when selecting the next token in a response.", default=0.95, show_default=True)
@click.option("-f", "--forgetful", help="Enable 'forgetful' mode - i.e. refresh the conversation after each response.", is_flag=True, default=False)
@main.command(short_help="Starts a chat with an LLM.")
def chat(
    config: Optional[str] = None,
    role: Optional[str] = None,
    system: Optional[str] = None,
    merge: bool = False,
    overrides: str = None,
    debug: bool = False,
    unsafe: bool = False,
    temperature: float=0.7,
    top_k: int=50,
    top_p: float=0.95,
    forgetful: bool = False
) -> None:
    """
    Runs an interactive chat in the command line.
    """
    configuration = get_configuration(
        config,
        overrides=overrides,
        merge=merge,
        debug=debug
    )
    from enfugue.diffusion.manager import DiffusionPipelineManager
    manager = DiffusionPipelineManager(configuration)

    with get_context(debug):
        click.echo(termcolor.colored("Loading language model. Say 'reset' at any time to start the conversation over. Use Ctrl+D to exit or say 'exit.'", "yellow"))
        if unsafe:
            click.echo(termcolor.colored("Safety is disengaged. Responses may be inappropriate or offensive, user discretion is advised.", "red"))
        with manager.conversation.converse(
            role=role,
            system=system,
            safe=not unsafe,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        ) as chat:
            try:
                click.echo(termcolor.colored("[assistant] {0}".format(chat()), "cyan")) # First message
                while True:
                    click.echo(
                        termcolor.colored(
                            "[assistant] {0}".format(
                                chat(input(termcolor.colored("[user] ", "green")))
                            ),
                            "cyan"
                        )
                    )
                    if forgetful:
                        chat("RESET")
            finally:
                click.echo("Goodbye!")

@click.option("-c", "--config", help="An optional path to a configuration file to use instead of the default.")
@click.option("-m", "--merge", is_flag=True, default=False, help="When set, merge the passed configuration with the default configuration instead of replacing it.")
@click.option("-o", "--overrides", help="an optional json object containing override configuration.")
@click.option("-d", "--debug", help="Enable debug logging.", is_flag=True, default=False)
@click.option("-b", "--browser", help="Open a browser window to the configured address when the server is ready to receive requests.", is_flag=True, default=False)
@main.command(short_help="Runs the server.")
def run(
    config: Optional[str] = None,
    merge: bool = False,
    overrides: str = None,
    debug: bool = False,
    browser: bool = False
) -> None:
    """
    Runs the server synchronously using cherrypy.
    """
    from enfugue.util.runner import EnfugueServerRunner
    configuration = get_configuration(
        config,
        overrides=overrides,
        merge=merge,
        debug=debug
    )
    runner = EnfugueServerRunner(configuration)

    try:
        runner.run(
            url_callback=lambda url: click.echo(f"Running ENFUGUE server accessible at {url}"),
            open_browser=browser
        )
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        click.echo(termcolor.colored(str(ex), "red"))
        if debug:
            click.echo(termcolor.colored(traceback.format_exc(), "red"))
    finally:
        click.echo("Goodbye!")

@click.argument("file_path")
@click.option("-d", "--debug", help="Enable debug logging.", is_flag=True, default=False)
@click.option("-s", "--seconds", help="The number of seconds to limit reading to.", default=300.0, show_default=True)
@click.option("-r", "--rate", help="The number of FFT samples to get per second.", default=8, show_default=True)
@click.option("-b", "--bands", help="The number of bands to limit to (linearly spaced).", default=512, show_default=True)
@click.option("-l", "--low-filter", help="A low pass, under which will be set to 0.", default=0.3, show_default=True)

@main.command(short_help="Analyzes the audio spectrum across an input file.")
def spectralyze(
    file_path: str,
    debug: bool = False,
    seconds: float = 300.0,
    rate: int = 8,
    bands: int = 512,
    low_filter: float = 0.0,
) -> None:
    """
    Analyzes frequencies in an audio file.
    """
    from enfugue.diffusion.util import Audio
    with get_context(debug):
        click.echo("Getting normalized frequencies and amplitudes from file.")
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        frequencies, amplitudes = Audio.from_file(file_path).get_normalized_frequencies(
            samples_per_second=rate,
            maximum_samples=rate*seconds,
            low_filter=low_filter
        )
        maxes = dict([
            (frequency, 0.0)
            for frequency in frequencies
        ])
        averages = dict([
            (frequency, 0.0)
            for frequency in frequencies
        ])
        max_freqs = []
        audio_frames = len(amplitudes)
        click.echo(f"Evaluating {audio_frames} spectral audio frames.")
        for i, frame in enumerate(amplitudes):
            frame_max_amplitude = 0
            frame_max_amplitude_index = 0
            for i, frequency in enumerate(frequencies):
                frame_amplitude = max(frame[i])
                if frame_amplitude > frame_max_amplitude:
                    frame_max_amplitude = frame_amplitude
                    frame_max_amplitude_index = i
                maxes[frequency] = max([maxes[frequency], frame_amplitude])
                averages[frequency] += frame_amplitude / audio_frames
            max_freqs.append(frequencies[frame_max_amplitude_index])

        from tabulate import tabulate
        headers = ["Second"] + [
            f"+{(i/rate):0.2f}s"
            for i in range(rate)
        ]
        table = []
        for i in range(0, audio_frames, rate):
            table += [[f"{(i/rate):0.2f}s"] + [
                f"{(max_freqs[j]):d}Hz"
                for j in range(i, i+rate)
                if j < audio_frames
            ]]

        click.echo(termcolor.colored("Frequency with Highest Amplitude over Time", "green"))
        click.echo(tabulate(table, headers=headers, tablefmt="grid"))

        min_freq = min(frequencies)
        max_freq = max(frequencies)
        freq_range = max_freq - min_freq
        freq_per_band = math.ceil(freq_range / bands)

        click.echo(termcolor.colored("Frequency Band Overall Analysis", "green"))
        for i in range(bands):
            band_min_freq = max_freq
            band_max_freq = 0
            band_max = 0
            band_averages = []
            band_freq_start = i * freq_per_band
            band_freq_end = ((i + 1) * freq_per_band) - 1
            for frequency in frequencies:
                if frequency >= band_freq_start and frequency < band_freq_end:
                    band_min_freq = min([band_min_freq, frequency])
                    band_max_freq = max([band_max_freq, frequency])
                    band_max = max([band_max, maxes[frequency]])
                    band_averages.append(averages[frequency])
                elif frequency >= band_freq_end:
                    break

            band_average = sum(band_averages) / len(band_averages)
            click.echo("{0}: {1}/{2}".format(
                termcolor.colored(f"{band_min_freq}-{band_max_freq} Hz", "cyan"),
                termcolor.colored(f"{band_max:0.3f} max", "green"),
                termcolor.colored(f"{band_average:0.3f} avg", "magenta")
            ))

try:
    main()
    sys.exit(0)
except Exception as ex:
    sys.stderr.write(f"{ex}\r\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    sys.exit(5)
