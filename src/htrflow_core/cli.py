import logging

import cowsay
import typer
import yaml

from htrflow_core.pipeline.pipeline import Pipeline
from htrflow_core.pipeline.steps import auto_import


app = typer.Typer(name="htrflow_core", add_completion=False, help="CLI inferface for htrflow_core's pipeline")


def setup_pipeline_logging(logfile: str, loglevel: str):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=time_format)
    logger = logging.getLogger()
    logger.setLevel(loglevel.upper())
    handler = logging.FileHandler(logfile, mode="w") if logfile else logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@app.command("pipeline-cli")
def main(
    pipeline: str = typer.Argument(..., help="Path to the pipeline configuration YAML file"),
    input_dirs: str = typer.Argument(..., help="Input directory or directories"),
    logfile: str = typer.Option(None, "--logfile", help="Log file path"),
    loglevel: str = typer.Option("info", "--loglevel", help="Logging level", case_sensitive=False),
):
    """Entrypoint for htrflow_core's pipeline."""
    setup_pipeline_logging(logfile, loglevel.upper())
    try:
        with open(pipeline, "r") as file:
            config = yaml.safe_load(file)

        pipe = Pipeline.from_config(config)

        volume = auto_import(input_dirs)

        typer.echo("Running Pipeline")
        volume = pipe.run(volume)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def test(msg: str = "Hello World"):
    """Test CLI with cowsay"""
    typer.echo(cowsay.get_output_string("cow", msg))


if __name__ == "__main__":
    app()
