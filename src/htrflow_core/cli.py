import logging
import socket
import time
from pathlib import Path
from typing import List

import cowsay
import typer
import yaml
from typing_extensions import Annotated

from htrflow_core.models import hf_utils
from htrflow_core.pipeline.pipeline import Pipeline
from htrflow_core.pipeline.steps import auto_import


app = typer.Typer(name="htrflow_core", add_completion=False, help="CLI inferface for htrflow_core's pipeline")


class HTRFLOWLoggingFormatter(logging.Formatter):
    """Logging formatter for HTRFLOW"""

    converter = time.gmtime

    def __init__(self):
        datefmt = "%Y-%m-%d %H:%M:%S"
        hostname = socket.gethostname()
        fmt = f"{hostname} - %(asctime)s UTC - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)


def setup_pipeline_logging(logfile: str, loglevel: str):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.setLevel(loglevel.upper())
    handler = logging.FileHandler(logfile, mode="w") if logfile else logging.StreamHandler()
    handler.setFormatter(HTRFLOWLoggingFormatter())
    logger.addHandler(handler)


def check_file_exists(file_path_str: str):
    """Ensure the path exists and is a file."""
    file_path = Path(file_path_str)
    if not file_path.exists() or not file_path.is_file():
        typer.echo(f"The file {file_path} does not exist or is not a valid file.")
        raise typer.Exit(code=1)
    return file_path_str


def check_folder_exists(folder_paths_str: List[str]):
    """Check each path exists and is a folder."""
    for folder_path_str in folder_paths_str:
        folder_path = Path(folder_path_str)
        if not folder_path.exists() or not folder_path.is_dir():
            typer.echo(f"The path {folder_path} does not exist or is not a folder.")
            raise typer.Exit(code=1)
    return folder_paths_str


def validate_logfile_extension(logfile: str):
    """Ensure the logfile string has a .log extension."""
    if logfile and not logfile.endswith(".log"):
        typer.echo(f"The logfile must have a .log extension. Provided: {logfile}")
        raise typer.Exit(code=1)
    return logfile


@app.command("pipeline")
def main(
    pipeline: Annotated[
        str, typer.Argument(..., help="Path to the pipeline config YAML file", callback=check_file_exists)
    ],
    input_dirs: Annotated[
        List[str], typer.Argument(..., help="Input directory or directories", callback=check_folder_exists)
    ],
    logfile: Annotated[
        str,
        typer.Option(help="Log file path", rich_help_panel="Secondary Arguments", callback=validate_logfile_extension),
    ] = None,
    loglevel: Annotated[
        str,
        typer.Option(help="Logging level", case_sensitive=False, rich_help_panel="Secondary Arguments"),
    ] = "info",
):
    """Entrypoint for htrflow_core's pipeline."""
    setup_pipeline_logging(logfile, loglevel.upper())
    try:
        with open(pipeline, "r") as file:
            config = yaml.safe_load(file)
        hf_utils.HF_CONFIG |= config.get("huggingface_config", {})
        pipe = Pipeline.from_config(config)

        volume = auto_import(input_dirs)

        typer.echo("Running Pipeline")
        volume = pipe.run(volume)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("cowsay")
def test(msg: Annotated[str, typer.Argument(help="Who Cow will greet")] = "Hello World"):
    """Test CLI with cowsay"""
    cow_msg = f"Hello {msg}"
    typer.echo(cowsay.get_output_string("cow", cow_msg))


if __name__ == "__main__":
    app()
