from typing import Sequence

from htrflow import progress
from htrflow.pipeline.steps import PipelineStep, init_step


class Pipeline:
    def __init__(self, steps: Sequence[PipelineStep]):
        self.steps = steps
        for step in self.steps:
            step.parent_pipeline = self

    @classmethod
    def from_config(self, config: dict[str, str]):
        """Init pipeline from config"""
        return Pipeline([init_step(step["step"], step.get("settings", {})) for step in config["steps"]])

    def run(self, document):
        """Run pipeline on document"""
        for step in self.steps:
            progress.update(document, status=str(step))
            document = step.run(document)
        progress.done(document)
        return document

    def metadata(self):
        return [step.metadata for step in self.steps if step.metadata]
