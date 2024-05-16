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
        processor: str | None = None,
        return_format: Literal["argmax", "softmax"] = "softmax",
        model_kwargs: dict | None = None,
        processor_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        model_kwargs = HF_CONFIG | (model_kwargs or {})
        processor_kwargs = HF_CONFIG | (processor_kwargs or {})

        self.return_format = return_format

        self.model = AutoModelForImageClassification.from_pretrained(model, **model_kwargs)

        self.model.to(self.set_device(self.device))
        logger.info(
            "Initialized DiT model from %s on device %s. Initialization parameters: %s",
            model,
            getattr(self.model, "device", "<device name unavailable>"),
            model_kwargs,
        )

        processor = processor or model
        self.processor = AutoImageProcessor.from_pretrained(processor, **processor_kwargs)

        logger.info("Initialized DiT processor from %s. Initialization parameters: %s", processor, processor_kwargs)

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

        return [
            Result(metadata=self.metadata, data=[{"classification": self._get_label(logits)}])
            for logits in batch_logits
        ]

    def _get_label(self, logits):
        if self.return_format == "argmax":
            predicted_class_idx = logits.argmax(-1).item()
            label_ = self.model.config.id2label[predicted_class_idx]

        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            label_ = {self.model.config.id2label[id]: prob.item() for id, prob in enumerate(probabilities)}
        return label_
