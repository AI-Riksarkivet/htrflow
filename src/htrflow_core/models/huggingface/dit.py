from os import PathLike

import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.torch_mixin import PytorchMixin
from htrflow_core.results import Result


# TODO: change model to model_path?


class DiT(BaseModel, PytorchMixin):
    def __init__(
        self,
        model: str | PathLike = "microsoft/dit-base-finetuned-rvlcdip",
        processor: str | PathLike = "microsoft/dit-base-finetuned-rvlcdip",
        *model_args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = AutoModelForImageClassification.from_pretrained(
            model, cache_dir=self.cache_dir, token=self.hf_token, *model_args
        )

        self.model.to(self.set_device(self.device))

        processor = processor or model

        self.processor = AutoImageProcessor.from_pretrained(processor, cache_dir=self.cache_dir, token=self.hf_token)

        self.metadata.update(
            {
                "model": str(model),
                "processor": str(processor),
                "framework": Framework.HuggingFace.value,
                "task": Task.ImageClassification.value,
                "device": self.device,
            }
        )

    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        inputs = self.processor(images, return_tensors="pt").to(self.model.device)

        logits = self.model(**inputs).logits
        predicted_class_idx = logits.argmax(-1).item()
        classification_label = self.model.config.id2label[predicted_class_idx]

        return [
            Result(image, metadata=self.metadata, data=[{"classification": classification_label}]) for image in images
        ]
