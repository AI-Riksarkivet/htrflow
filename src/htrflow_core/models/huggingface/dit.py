import logging
from typing import Literal

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import HF_CONFIG
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.results import Result


logger = logging.getLogger(__name__)


class DiT(BaseModel, PytorchMixin):
    def __init__(
        self,
        model: str = "microsoft/dit-base-finetuned-rvlcdip",
        processor: str = "microsoft/dit-base-finetuned-rvlcdip",
        return_format: Literal["argmax", "softmax"] = "softmax",
        *model_args,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.return_format = return_format

        self.model = AutoModelForImageClassification.from_pretrained(model, *model_args, **HF_CONFIG)

        self.model.to(self.set_device(self.device))
        logger.info(
            "Initialized DiT model from %s on device %s",
            model,
            getattr(self.model, "device", "<device name unavailable>"),
        )

        processor = processor or model

        self.processor = AutoImageProcessor.from_pretrained(processor, **HF_CONFIG)

        logger.info("Initialized DiT processor from %s", processor)

        self.metadata.update(
            {
                "model": str(model),
                "processor": str(processor),
                "framework": Framework.HuggingFace.value,
                "task": Task.ImageClassification.value,
                "device": self.device_id,
                "return_format": self.return_format,
            }
        )

    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        inputs = self.processor(images, return_tensors="pt").pixel_values

        with torch.no_grad():
            batch_logits = self.model(inputs.to(self.model.device)).logits

        return self._create_classification_results(images, batch_logits)

    def _create_classification_results(self, images, batch_logits):
        classification_labels = [self._label_based_on_return_format(logits) for logits in batch_logits]

        return [
            Result(image, metadata=self.metadata, data=[{"classification": label}])
            for image, label in zip(images, classification_labels)
        ]

    def _label_based_on_return_format(self, logits):
        if self.return_format == "argmax":
            predicted_class_idx = logits.argmax(-1).item()
            label_ = self.model.config.id2label[predicted_class_idx]

        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            label_ = {self.model.config.id2label[id]: prob.item() for id, prob in enumerate(probabilities)}
        return label_
