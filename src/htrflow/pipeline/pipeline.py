import logging
from typing import Sequence

from htrflow.pipeline.steps import PipelineStep, init_step


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, steps: Sequence[PipelineStep]):
        self.steps = steps
        for step in self.steps:
            step.parent_pipeline = self

    @classmethod
    def from_config(self, config: dict[str, str]):
        """Init pipeline from config"""
        return Pipeline([init_step(step["step"], step.get("settings", {})) for step in config["steps"]])

    def run(self, collection):
        """Run pipeline on collection"""
        for i, step in enumerate(self.steps):
            step_name = f"{step} (step {i + 1} / {len(self.steps)})"
            logger.info("Running step %s", step_name)
            collection = step.run(collection)

        return collection

    def metadata(self):
        return [step.metadata for step in self.steps if step.metadata]
