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


app = typer.Typer(
    name="htrflow",
    add_completion=False,
    help="CLI inferface for htrflow",
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
        list[str],
        typer.Argument(help="Paths to input images. May be paths to directories of images or paths to single images."),
    ],
    logfile: Annotated[
        str, typer.Option(help="Where to write logs to. If not provided, logs will be printed to the standard output.")
    ] = None,
    loglevel: Annotated[LogLevel, typer.Option(help="Loglevel", case_sensitive=False)] = LogLevel.info,
    backup: Annotated[bool, typer.Option(help="Save a pickled backup after each pipeline step.")] = False,
    batch_output: Annotated[
        int | None, typer.Option(help="Write continuous output in batches of this size (number of images).")
    ] = None,
):
    """Run a HTRFlow pipeline"""

    # Slow imports! Only import after all CLI arguments have been resolved.
    from htrflow.models import hf_utils
    from htrflow.pipeline.pipeline import Pipeline
    from htrflow.pipeline.steps import auto_import

    logger = setup_pipeline_logging(logfile, loglevel)

    with open(pipeline, "r") as file:
        config = yaml.safe_load(file)

    hf_utils.HF_CONFIG |= config.get("huggingface_config", {})
    pipe = Pipeline.from_config(config)
    pipe.do_backup = backup

    tic = time.time()
    collections = auto_import(inputs, max_size=batch_output)
    processed = []
    for collection in collections:
        if "labels" in config:
            collection.set_label_format(**config["labels"])
        processed.append(pipe.run(collection))
    toc = time.time()

    total_time = toc - tic
    n_pages = sum(len(collection.pages) for collection in processed)
    logger.info(
        "Processed %d pages in %d seconds (average %.3f seconds per page)",
        n_pages,
        total_time,
        total_time / n_pages if n_pages > 0 else -1.0,
    )

    return processed


@app.command("evaluate")
def run_evaluation(
    gt: Annotated[
        str,
        typer.Argument(
            help="Path to directory with ground truth files. Should have two subdirectories `images` and `xmls`."
        ),
    ],
    candidates: Annotated[
        list[str], typer.Argument(help="Paths to pipelines or directories containing already generated Page XMLs.")
    ],
):
    """
    Evaluate HTR transcriptions against ground truth
    """

    from htrflow.evaluate import evaluate
    from htrflow.pipeline.steps import join_collections

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
            collections = run_pipeline(pipe, images, logfile=os.path.join(pipeline_dir, "htrflow.log"))
            collection = join_collections(collections)
            collection.label = pipeline_name

            # Save PageXMLs in `run_dir` and add the path to the XMLs to
            # the candidates
            collection.save(run_dir, "page")
            candidates.append(os.path.join(run_dir, collection.label))

    df = evaluate(gt, *candidates)
    df.to_csv(os.path.join(run_dir, "evaluation_results.csv"))


if __name__ == "__main__":
    app()
