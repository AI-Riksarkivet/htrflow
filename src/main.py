"""
Entrypoint for HTRFLOW

Usage:
> python src/main.py <pipeline.yaml> <input_directory>

Example:
> python src/main.py data/pipelines/demo.yaml data/demo_images/A0068699
"""
import logging
import sys

import yaml

from htrflow_core.pipeline.pipeline import Pipeline


logging.basicConfig(filename='htrflow.log', level=logging.INFO, filemode='w')

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    pipe = Pipeline.from_config(config)
    pipe.run("data/demo_images/A0068699")
