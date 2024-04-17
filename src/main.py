"""
Entrypoint for HTRFLOW

Usage:
> python src/main.py <pipeline.yaml> <input_directory>

Example:
> python src/main.py data/pipelines/demo.yaml data/demo_images/A0068699
"""
import argparse
import logging

import yaml

from htrflow_core.pipeline.pipeline import Pipeline
from htrflow_core.pipeline.steps import auto_import


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HTRFLOW")
    parser.add_argument("pipeline")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="outputs")
    parser.add_argument("--logfile", default="htrflow.log")
    parser.add_argument("--loglevel", choices=["debug", "info", "warning", "error"], default="info")
    args = parser.parse_args()

    logging.basicConfig(filename=args.logfile, level=args.loglevel.upper(), filemode="w")

    with open(args.pipeline, "r") as f:
        config = yaml.safe_load(f)

    pipe = Pipeline.from_config(config)
    volume = auto_import(args.input)
    volume = pipe.run(volume)
