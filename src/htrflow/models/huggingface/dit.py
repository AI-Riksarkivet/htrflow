import logging
from typing import Literal

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import AutoImageProcessor, AutoModelForImageClassification

from htrflow.models.base_model import BaseModel
from htrflow.models.hf_utils import HF_CONFIG
from htrflow.results import Result


logger = logging.getLogger(__name__)


class DiT(BaseModel):
    """
    HTRFLOW adapter of DiT for image classification.

    Uses huggingface's implementation of DiT. For further
    information about the model, see
    https://huggingface.co/docs/transformers/model_doc/dit.
    """

    def __init__(
        self,
        model: str,
        processor: str | None = None,
        model_kwargs: dict | None = None,
        processor_kwargs: dict | None = None,
        device: str | None = None,
    ):
        """Initialize a DiT model

        Arguments:
            model: Path or name of pretrained AutoModelForImageClassification.
            processor: Optional path or name of a pretrained AutoImageProcessor.
                If not given, the given `model` is used.
            model_kwargs: Model initialization kwargs that are forwarded to
                AutoModelForImageClassification.from_pretrained().
            processor_kwargs: Processor initialization kwargs that are forwarded
                to AutoImageProcessor.from_pretrained().
            kwargs: Additional kwargs that are forwarded to BaseModel's __init__.
        """
        super().__init__(device)

        # Initialize model
        model_kwargs = HF_CONFIG | (model_kwargs or {})
        self.model = AutoModelForImageClassification.from_pretrained(model, **model_kwargs)
        self.model.to(self.device)
        logger.info("Initialized DiT model from %s on device %s.", model, self.device)

        # Initialize processor
        processor = processor or model
        processor_kwargs = HF_CONFIG | (processor_kwargs or {})
        self.processor = AutoImageProcessor.from_pretrained(processor, **processor_kwargs)
        logger.info(
            "Initialized DiT processor from %s. Initialization parameters: %s",
            processor,
            processor_kwargs,
        )

        self.metadata.update(
            {
                "model": model,
                "model_version": model_info(model).sha,
                "processor": processor,
                "processor_version": model_info(processor).sha,
            }
        )

    def _predict(
        self,
        images: list[np.ndarray],
        return_format: Literal["argmax", "softmax"] = "softmax",
    ) -> list[Result]:
        """Perform inference on `images`

        Arguments:
            images: List of input images.
            return_format: Decides the format of the output. Options are:
                - "softmax": returns the confidence scores for each class
                    label and image. Default.
                - "argmax": returns the most probable class label for each
                    image.
        """
        inputs = self.processor(images, return_tensors="pt").pixel_values

        with torch.no_grad():
            batch_logits = self.model(inputs.to(self.model.device)).logits

        return [
            Result(
                metadata=self.metadata,
                data=[{"classification": self._get_labels(logits, return_format)}],
            )
            for logits in batch_logits
        ]

    def _get_labels(self, logits, return_format):
        if return_format == "argmax":
            predicted_class_idx = logits.argmax(-1).item()
            label_ = self.model.config.id2label[predicted_class_idx]

        else:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            label_ = {self.model.config.id2label[id]: prob.item() for id, prob in enumerate(probabilities)}
        return label_
