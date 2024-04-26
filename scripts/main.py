"""
Entrypoint for HTRFLOW

Usage:
  python src/main.py <pipeline.yaml> <input_directory> [options]

Options:
  --logfile          Specify the log file.
  --loglevel         Set the logging level (debug, info, warning, error). Default is "info".

Example:
  python src/main.py data/pipelines/demo.yaml data/demo_images/A0068699
"""

import argparse
import logging

import yaml

from htrflow_core.pipeline.pipeline import Pipeline
from htrflow_core.pipeline.steps import auto_import


def setup_pipeline_logging(logfile, loglevel):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=time_format)
    logger = logging.getLogger()
    logger.setLevel(loglevel.upper())
    handler = logging.FileHandler(logfile, mode="w") if logfile else logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(args):
    setup_pipeline_logging(args.logfile, args.loglevel)
    with open(args.pipeline, "r") as file:
        config = yaml.safe_load(file)
    pipe = Pipeline.from_config(config)
    volume = auto_import(args.input)
    volume.set_label_format(**config["labels"])
    volume = pipe.run(volume, start=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint for HTRFLOW")
    parser.add_argument("pipeline", help="Path to the pipeline configuration YAML file")
    parser.add_argument("input", nargs="+", help="Input directory or files")
    parser.add_argument("--logfile", help="Log file path")
    parser.add_argument(
        "--loglevel", choices=["debug", "info", "warning", "error"], default="info", help="Logging level"
    )
    parser.add_argument("--resume", type=int, default=0, help="Resume pipeline from this step")
    args = parser.parse_args()

    main(args)
