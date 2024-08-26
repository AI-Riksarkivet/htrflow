import logging
import os
import shutil
import socket
import time
from datetime import datetime
from enum import Enum

import typer
import yaml
from typing_extensions import Annotated

from htrflow.evaluate import evaluate
from htrflow.models import hf_utils
from htrflow.pipeline.pipeline import Pipeline
from htrflow.pipeline.steps import auto_import


app = typer.Typer(
    name="htrflow",
    add_completion=False,
    help="CLI inferface for htrflow",
)


# This is needed in order for Typer to print the choices in
# the generated CLI help.
class LogLevel(Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class HTRFLOWLoggingFormatter(logging.Formatter):
    """Logging formatter for HTRFLOW"""

    converter = time.gmtime

    def __init__(self):
        datefmt = "%Y-%m-%d %H:%M:%S"
        hostname = socket.gethostname()
        fmt = f"{hostname} - %(asctime)s UTC - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)


def setup_pipeline_logging(logfile: str | None, loglevel: LogLevel):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.setLevel(loglevel.value.upper())
    if logfile is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(logfile, mode="w")
    handler.setFormatter(HTRFLOWLoggingFormatter())
    logger.addHandler(handler)


@app.command("pipeline")
def run_pipeline(
    pipeline: Annotated[
        str, typer.Argument(help="Path to a HTRFlow pipeline YAML file")
    ],
    inputs: Annotated[
        list[str],
        typer.Argument(
            help="Paths to input images. May be paths to directories of images or paths to single images."
        ),
    ],
    logfile: Annotated[
        str,
        typer.Option(
            help="Where to write logs to. If not provided, logs will be printed to the standard output."
        ),
    ] = None,
    loglevel: Annotated[
        LogLevel, typer.Option(help="Loglevel", case_sensitive=False)
    ] = LogLevel.info,
):
    """Run a HTRFlow pipeline"""
    setup_pipeline_logging(logfile, loglevel)

    with open(pipeline, "r") as file:
        config = yaml.safe_load(file)
    hf_utils.HF_CONFIG |= config.get("huggingface_config", {})
    pipe = Pipeline.from_config(config)

    volume = auto_import(inputs)

    if "labels" in config:
        volume.set_label_format(**config["labels"])

    print(f"Running pipeline {pipeline}")
    return pipe.run(volume)


@app.command("evaluate")
def run_evaluation(
    gt: Annotated[
        str,
        typer.Argument(
            help="Path to directory with ground truth files. Should have two subdirectories `images` and `xmls`."
        ),
    ],
    candidates: Annotated[
        list[str],
        typer.Argument(
            help="Paths to pipelines or directories containing already generated Page XMLs."
        ),
    ],
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
            collection = run_pipeline(
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
