import logging
from typing import Literal

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.torch_mixin import PytorchMixin
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

        self.model = AutoModelForImageClassification.from_pretrained(
            model, cache_dir=self.cache_dir, token=self.hf_token, *model_args
        )

        self.model.to(self.set_device(self.device))
        logger.info(f"Model loaded ({self.device}) from {model}.")

        processor = processor or model

        self.processor = AutoImageProcessor.from_pretrained(processor, cache_dir=self.cache_dir, token=self.hf_token)

        logger.info(f"Processor loaded from {processor}.")

        self.metadata.update(
            {
                "model": str(model),
                "processor": str(processor),
                "framework": Framework.HuggingFace.value,
                "task": Task.ImageClassification.value,
                "device": self.device,
                "return_format": self.return_format,
            }
        )

    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        inputs = self.processor(images, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        if self.return_format == "argmax":
            predicted_class_idx = logits.argmax(-1).item()
            classification_label = self.model.config.id2label[predicted_class_idx]
        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            label_probabilities = {
                self.model.config.id2label[i]: probabilities[0][i].item() for i in range(len(probabilities[0]))
            }

            classification_label = label_probabilities

        logger.info(f"Prediction complete. {self.return_format}: classification: {classification_label}.")

        return [
            Result(image, metadata=self.metadata, data=[{"classification": classification_label}]) for image in images
        ]
