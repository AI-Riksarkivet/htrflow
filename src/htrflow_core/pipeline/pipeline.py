import logging
from typing import Sequence

from htrflow_core.pipeline.steps import PipelineStep, auto_import, init_step


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, steps: Sequence[PipelineStep]):
        self.steps = steps
        validate(self)

    @classmethod
    def from_config(self, config: dict[str, str]):
        """Init pipeline from config"""
        return Pipeline([init_step(step) for step in config["steps"]])

    def run(self, volume):
        """Run pipeline on volume"""
        volume = auto_import(volume)
        for i, step in enumerate(self.steps):
            logger.info("Running step %s (step %d/%d)", step, i+1, len(self.steps))
            volume = step.run(volume)
        return volume

    def metadata(self):
        return [step.metadata for step in self.steps]


def validate(pipeline: Pipeline):
    steps = [step.__class__ for step in pipeline.steps]
    for i, step in enumerate(steps):
        for req_step in step.requires:
            if req_step not in steps[:i]:
                raise RuntimeError(f"Not valid pipeline: {step.__name__} must be preceded by {req_step.__name__}")
            logger.info("Validating pipeline: %s is preceded by %s - OK", step.__name__, req_step.__name__)
    logger.info("Pipeline passed validation")
