import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Iterable

import typer
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


class OutputFormat(str, Enum):
    alto = "alto"
    page = "page"
    plaintext = "txt"
    json = "json"


class HTRFLOWLoggingFormatter(logging.Formatter):
    """Logging formatter for HTRFLOW"""

    converter = time.gmtime

    def __init__(self):
        datefmt = "%Y-%m-%d %H:%M:%S"
        hostname = socket.gethostname()
        fmt = f"{hostname} - %(asctime)s UTC - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)


def setup_pipeline_logging(logfile: str, loglevel: LogLevel):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.setLevel(loglevel.value.upper())
    handler = logging.FileHandler(logfile, mode="w")
    handler.setFormatter(HTRFLOWLoggingFormatter())
    logger.addHandler(handler)
    return logger


@app.callback()
def callback():
    pass


@app.command()
def pipeline(
    pipeline: Annotated[str, typer.Argument(help="Path to a HTRflow pipeline YAML file")],
    inputs: Annotated[
        list[str] | None,
        typer.Argument(help="Paths to input images. May be paths to directories of images or paths to single images."),
    ] = None,
    logfile: Annotated[str, typer.Option(help="Logfile")] = "htrflow.log",
    loglevel: Annotated[LogLevel, typer.Option(help="Loglevel", case_sensitive=False)] = LogLevel.info,
    output: Annotated[
        str | None,
        typer.Option(
            help=(
                "Output directory. Adds an extra `Export` step to the end of the given pipeline. Uses the format "
                "given by --output_format, or plaintext if not given."
            )
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat | None,
        typer.Option(
            help=(
                "Output format. Adds an extra `Export` step to the end of the given pipeline. Writes to the directory "
                "given by --output, or 'outputs' if not given."
            )
        ),
    ] = None,
    inputs_file: Annotated[
        str | None,
        typer.Option(
            help="A text file containing newline-separated paths to input images. Requires INPUTS to be empty."
        ),
    ] = None,
    workers: Annotated[int, typer.Option(help="Number of concurrent workers")] = 1,
):
    """Run a HTRflow pipeline"""

    logger = setup_pipeline_logging(logfile, loglevel)
    inputs = get_inputs(inputs, inputs_file)

    # Slow imports! Only import after all CLI arguments have been resolved.
    from htrflow.pipeline.pipeline import Pipeline
    from htrflow.pipeline.steps import Export, auto_import

    pipeline = Pipeline.from_config(pipeline)

    if output or output_format:
        output = output or "outputs"
        output_format = output_format or "txt"
        pipeline.steps.append(Export(output, output_format))

    tic = time.time()
    n_pages = 0

    with ThreadPoolExecutor(workers) as executor:
        for document in auto_import(inputs):
            executor.submit(pipeline.run, document)
            n_pages += 1
    toc = time.time()

    total_time = toc - tic
    logger.info(
        "Processed %d pages in %d seconds (average %.3f seconds per page)",
        n_pages,
        total_time,
        total_time / n_pages if n_pages > 0 else -1.0,
    )


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
