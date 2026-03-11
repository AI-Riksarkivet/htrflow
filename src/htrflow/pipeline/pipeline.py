import yaml

from htrflow import progress
from htrflow.pipeline.steps import PipelineConfig, PipelineStep, init_step


class Pipeline:
    steps: list[PipelineStep]

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    @classmethod
    def from_config(cls, path: str) -> "Pipeline":
        """
        Create a pipeline from a YAML config file.

        Arguments:
            path: Path to YAML config.
        """
        with open(path, "r") as file:
            config = yaml.safe_load(file)

        config = PipelineConfig(**config)
        steps = list(map(init_step, config.steps))
        return cls(steps)

    def run(self, document):
        """Run pipeline on document"""
        for step in self.steps:
            progress.step(document, step=step)
            document = step.run(document)
        progress.done(document)
        return document
