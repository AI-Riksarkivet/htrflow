import yaml

from htrflow import progress
from htrflow.pipeline.steps import PipelineConfig, PipelineStep, init_step


class Pipeline:
    steps: list[PipelineStep]

    def __init__(self, path: str):

        with open(path, "r") as file:
            config = yaml.safe_load(file)

        config = PipelineConfig(**config)
        self.steps = []
        for step in map(init_step, config.steps):
            step.parent_pipeline = self  # TODO: solve metadata export in a better way than this
            self.steps.append(step)

    def run(self, document):
        """Run pipeline on document"""
        for step in self.steps:
            progress.update(document, status=str(step))
            document = step.run(document)
        progress.done(document)
        return document

    def metadata(self):
        return [step.metadata for step in self.steps if step.metadata]
