import torch

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from htrflow.inferencers.base_inferencer import BaseInferencer


class UltralyticsInferencer(BaseInferencer):
    def __init__(self, model: Model):
        self._model = model

    def preprocess(self, inputs):
        return inputs

    def predict(self, inputs, *args, **kwargs):
        inputs = self.preprocess(inputs)
        results = self._model(inputs, *args, **kwargs)
        return self.postprocess(results)

    def postprocess(self, results: list[Results]) -> list[list[tuple]]:
        """Map list of Results objects to lists of (x_min, y_min, x_max, y_max, confidence_score) tuples"""
        return [self._results_to_tuples(res) for res in results]

    def _results_to_tuples(self, results: Results) -> list[tuple]:
        return [tuple(xyxyc) for xyxyc in torch.cat((results.boxes.xyxy, results.boxes.conf.unsqueeze(1)), 1).tolist()]
