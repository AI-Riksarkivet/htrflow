import logging
import os
import shutil
import socket
import time
from datetime import datetime
from enum import Enum
from typing import Iterable

import typer
import yaml
from typing_extensions import Annotated


app = typer.Typer(
    name="htrflow", add_completion=False, help="CLI inferface for htrflow", pretty_exceptions_enable=False
)


# This is needed in order for Typer to print the choices in
# the generated CLI help.
class LogLevel(str, Enum):
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
    return logger


@app.command("pipeline")
def run_pipeline(
    pipeline: Annotated[str, typer.Argument(help="Path to a HTRFlow pipeline YAML file")],
    inputs: Annotated[
        list[str] | None,
        typer.Argument(help="Paths to input images. May be paths to directories of images or paths to single images."),
    ] = None,
    logfile: Annotated[
        str,
        typer.Option(help="Where to write logs to. If not provided, logs will be printed to the standard output."),
    ] = None,
    loglevel: Annotated[LogLevel, typer.Option(help="Loglevel", case_sensitive=False)] = LogLevel.info,
    backup: Annotated[bool, typer.Option(help="Save a pickled backup after each pipeline step.")] = False,
    batch_output: Annotated[
        int | None,
        typer.Option(help="Write continuous output in batches of this size (number of images)."),
    ] = 1,
    label: Annotated[
        str | None,
        typer.Option(help="Collection label"),
    ] = None,
    inputs_file: Annotated[
        str | None,
        typer.Option(
            help="A text file containing newline-separated paths to input images. Requires INPUTS to be empty."
        ),
    ] = None,
):
    """Run a HTRFlow pipeline"""

    logger = setup_pipeline_logging(logfile, loglevel)
    inputs = get_inputs(inputs, inputs_file)

    # Slow imports! Only import after all CLI arguments have been resolved.
    from htrflow.models import hf_utils
    from htrflow.pipeline.pipeline import Pipeline
    from htrflow.pipeline.steps import auto_import

    if isinstance(pipeline, Pipeline):
        pipe = pipeline
        config = {}
    else:
        with open(pipeline, "r") as file:
            config = yaml.safe_load(file)
        pipe = Pipeline.from_config(config)

    hf_utils.HF_CONFIG |= config.get("huggingface_config", {})
    pipe.do_backup = backup

    tic = time.time()
    collections = auto_import(inputs, max_size=batch_output)
    n_pages = 0
    for collection in collections:
        if "labels" in config:
            collection.set_label_format(**config["labels"])
        if label:
            collection.label = label
        collection = pipe.run(collection)
        n_pages += len(collection.pages)
    toc = time.time()

    total_time = toc - tic
    logger.info(
        "Processed %d pages in %d seconds (average %.3f seconds per page)",
        n_pages,
        total_time,
        total_time / n_pages if n_pages > 0 else -1.0,
    )


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
        typer.Argument(help="Paths to pipelines or directories containing already generated Page XMLs."),
    ],
):
    """
    Evaluate HTR transcriptions against ground truth
    """

    from htrflow.evaluate import evaluate
    from htrflow.pipeline.pipeline import Pipeline
    from htrflow.pipeline.steps import Export

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

            with open(pipe, "r") as file:
                pipeline = Pipeline.from_config(yaml.safe_load(file))
            pipeline.steps.append(Export(run_dir, "page"))

            # Run the pipeline
            run_pipeline(pipeline, images, logfile=os.path.join(pipeline_dir, "htrflow.log"), label=pipeline_name)
            candidates.append(os.path.join(run_dir, pipeline_name))

    df = evaluate(gt, *candidates)
    df.to_csv(os.path.join(run_dir, "evaluation_results.csv"))


def get_inputs(inputs: list[str] | None, inputs_file: str | None) -> Iterable[str]:
    """
    Get inputs from the CLI arguments `inputs` and `inputs_file`

    The HTRflow CLI accepts inputs in two formats: Either as a list of paths,
    given as positional arguments:

        $ htrflow pipeline pipeline.yaml image1.jpg image2.jpg image3.jpg

    or as a path to a text file containing the same information. In this case,
    the file is given with the argument --inputs-file:

        $ htrflow pipeline pipeline.yaml --inputs-file=inputs.txt

    This function parses the two arguments and returns a list of input paths.

    Arguments:
        inputs: The list of inputs given as positional arguments
        inputs_file: Path to a text file containing inputs

    Raises:
        typer.BadParameter if both `inputs` and `inputs_file` are None, or
        if both are given.
    """
    if inputs is not None:
        if inputs_file:
            raise typer.BadParameter(
                f"Please provide only one of INPUTS and --inputs-file (got INPUTS={inputs} and --inputs-file={inputs_file})"
            )
        return inputs

    if inputs_file:
        with open(inputs_file, "r") as f:
            inputs = map(str.strip, f.readlines())
        return inputs

    raise typer.BadParameter("Missing input files. Please provide either INPUTS or --inputs-file.")


if __name__ == "__main__":
    app()
