from os import PathLike
from typing import Optional

import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.torch_mixin import PytorchDeviceMixin
from htrflow_core.results import Result


class DiT(BaseModel, PytorchDeviceMixin):
    def __init__(
        self,
        model: str | PathLike = "microsoft/dit-base-finetuned-rvlcdip",
        processor: str | PathLike = "microsoft/dit-base-finetuned-rvlcdip",
        device: Optional[str] = None,
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.model = AutoModelForImageClassification.from_pretrained(model, cache_dir=cache_dir, token=hf_token).to(
            self.set_device(device)
        )

        if processor is None:
            processor = model
        self.processor = AutoImageProcessor.from_pretrained(processor, cache_dir=cache_dir, token=hf_token)

        self.metadata = {
            "model": str(model),
            "processor": str(processor),
        }

    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        inputs = self.processor(images, return_tensors="pt").to(self.model.device)

        logits = self.model(**inputs).logits
        predicted_class_idx = logits.argmax(-1).item()
        classification_label = self.model.config.id2label[predicted_class_idx]

        return [
            Result(image, metadata=self.metadata, data=[{"classification": classification_label}]) for image in images
        ]


if __name__ == "__main__":
    url = "https://github.com/Swedish-National-Archives-AI-lab/htrflow_core/blob/a1b4b31f9a8b7c658a26e0e665eb536a0d757c45/data/demo_image.jpg?raw=true"

    model = DiT()
    results = model([url] * 10)

    print(results[0])
