import logging
from typing import Sequence

from htrflow_core.pipeline.steps import PipelineStep, auto_import, init_step
from htrflow_core.serialization import pickle_collection


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, steps: Sequence[PipelineStep]):
        self.steps = steps
        self.pickle_path = None
        for step in self.steps:
            step.parent_pipeline = self

    @classmethod
    def from_config(self, config: dict[str, str]):
        """Init pipeline from config"""
        return Pipeline([init_step(step["step"], step.get("settings", {})) for step in config["steps"]])

    def run(self, collection, start=0):
        """Run pipeline on collection"""
        collection = auto_import(collection)
        for i, step in enumerate(self.steps[start:]):
            step_name = f"{step} (step {start+i+1} / {len(self.steps)})"
            logger.info("Running step %s", step_name)
            try:
                collection = step.run(collection)
            except Exception:
                logger.error(
                    "Pipeline failed on step %s. A backup collection is saved at %s", step_name, self.pickle_path
                )
                raise
            self.pickle_path = pickle_collection(collection)
        return collection

    def metadata(self):
        return [step.metadata for step in self.steps if step.metadata]
