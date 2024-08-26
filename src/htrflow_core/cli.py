import logging
import os
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import List

import typer
import yaml
from typing_extensions import Annotated

from htrflow_core.evaluate import evaluate
from htrflow_core.models import hf_utils
from htrflow_core.pipeline.pipeline import Pipeline
from htrflow_core.pipeline.steps import auto_import


app = typer.Typer(
    name="htrflow_core",
    add_completion=False,
    help="CLI inferface for htrflow_core's pipeline",
)


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
    handler = (
        logging.FileHandler(logfile, mode="w") if logfile else logging.StreamHandler()
    )
    handler.setFormatter(HTRFLOWLoggingFormatter())
    logger.addHandler(handler)


def check_file_exists(file_path_str: str):
    """Ensure the path exists and is a file."""
    file_path = Path(file_path_str)
    if not file_path.exists() or not file_path.is_file():
        typer.echo(f"The file {file_path} does not exist or is not a valid file.")
        raise typer.Exit(code=1)
    return file_path_str


def validate_logfile_extension(logfile: str):
    """Ensure the logfile string has a .log extension."""
    if logfile and not logfile.endswith(".log"):
        typer.echo(f"The logfile must have a .log extension. Provided: {logfile}")
        raise typer.Exit(code=1)
    return logfile


@app.command("pipeline")
def main(
    pipeline: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to the pipeline config YAML file",
            callback=check_file_exists,
        ),
    ],
    inputs: Annotated[
        List[str],
        typer.Argument(
            ...,
            help="Input path(s) pointing to images or directories contatining images",
        ),
    ],
    logfile: Annotated[
        str,
        typer.Option(
            help="Log file path",
            rich_help_panel="Secondary Arguments",
            callback=validate_logfile_extension,
        ),
    ] = None,
    loglevel: Annotated[
        str,
        typer.Option(
            help="Logging level",
            case_sensitive=False,
            rich_help_panel="Secondary Arguments",
        ),
    ] = "info",
):
    """Entrypoint for htrflow_core's pipeline."""
    setup_pipeline_logging(logfile, loglevel.upper())

    with open(pipeline, "r") as file:
        config = yaml.safe_load(file)
    hf_utils.HF_CONFIG |= config.get("huggingface_config", {})
    pipe = Pipeline.from_config(config)

    volume = auto_import(inputs)

    if "labels" in config:
        volume.set_label_format(**config["labels"])
    typer.echo("Running Pipeline")
    volume = pipe.run(volume)
    return volume


@app.command("cowsay")
def test(msg: Annotated[str, typer.Argument(help="Who Cow will greet")] = "Hello World"):
    """Test CLI with cowsay"""
    cow_msg = f"Hello {msg}"
    typer.echo(cowsay.get_output_string("cow", cow_msg))


@app.command("evaluate")
def run_evaluation(
    gt: Annotated[
        str, typer.Argument(help="Path to directory with ground truth files. Should have two subdirectories `images` and `xmls`.")
    ],
    candidates: Annotated[
        list[str], typer.Argument(help="Paths to pipelines or directories containing already generated Page XMLs.")
    ]
):
    """
    Evaluate HTR transcriptions against ground truth
    """
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("evaluation", run_name)
    os.makedirs(run_dir)

    # inputs are pipelines -> run the pipelines before evaluation
    if all(os.path.isfile(candidate) for candidate in candidates):
        images = os.path.join(gt, "images")
        pipelines = candidates
        candidates = []
        for i, pipe in enumerate(pipelines):

            # Create a directory under `run_dir` to save pipeline results,
            # logs and a copy of the pipeline yaml to.
            pipeline_name = f"pipeline{i}_{os.path.splitext(os.path.basename(pipe))[0]}"
            pipeline_dir = os.path.join(run_dir, pipeline_name)
            os.mkdir(pipeline_dir)
            shutil.copy(pipe, os.path.join(pipeline_dir, pipeline_name + ".yaml"))

            # Run the pipeline
            collection = pipeline(
                pipe, images, logfile=os.path.join(pipeline_dir, "htrflow.log")
            )
            collection.label = pipeline_name

            # Save PageXMLs in `run_dir` and add the path to the XMLs to
            # the candidates
            collection.save(run_dir, "page")
            candidates.append(os.path.join(run_dir, collection.label))

    df = evaluate(gt, *candidates)
    df.to_csv(os.path.join(run_dir, "evaluation_results.csv"))


if __name__ == "__main__":
    app()
