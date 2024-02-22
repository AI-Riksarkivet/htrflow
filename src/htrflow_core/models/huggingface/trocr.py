from typing import Iterable

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from htrflow_core.models.base_model import BaseModel
from htrflow_core.results import RecognitionResult


class TrOCR(BaseModel):
    default_generation_kwargs = {
        "num_beams": 4,
    }

    def __init__(
        self,
        model_source: str = "microsoft/trocr-small-handwritten",
        processor_source: str = "microsoft/trocr-base-handwritten",
    ):
        """Initialize a TrOCR model

        Arguments:
            model_source: Path or name of pretrained VisisonEncoderDeocderModel.
                Defaults to 'microsoft/trocr-small-handwritten'.
            processor_source: Path or name of pretrained TrOCRProcessor.
                Defaults to 'microsoft/trocr-base-handwritten'.
        """
        self.model = VisionEncoderDecoderModel.from_pretrained(model_source)

        if processor_source is None:
            processor_source = model_source
        self.processor = TrOCRProcessor.from_pretrained(processor_source)

        self.metadata = {
            "model": self.model.name_or_path,
            "processor": self.processor.to_dict(),
        }

    def predict(self, images: Iterable[np.ndarray], **generation_kwargs) -> Iterable[RecognitionResult]:
        """Perform inference on `images`

        Uses beam search with 4 beams by default, this can be altered
        by setting `num_beams` in generation_kwargs.

        Arguments:
            images: Input images.
            **generation_kwargs: Optional keyword arguments that are
                forwarded to the model's .generate() method.

        Returns:
            The predicted texts and their beam scores.
        """
        # Prepare generation keyword arguments
        generation_kwargs = self._prepare_generation_kwargs(**generation_kwargs)
        metadata = self.metadata | {"generation_args": generation_kwargs}

        # Run inference
        model_inputs = self.processor(images, return_tensors="pt").pixel_values
        model_outputs = self.model.generate(model_inputs, **generation_kwargs)

        # Prepare output
        texts = self.processor.batch_decode(model_outputs.sequences, skip_special_tokens=True)
        scores = model_outputs.sequences_scores.tolist()
        # `texts` and `scores` are flattened lists so we need to iterate
        # over them in steps to ensure that the list of results correspond
        # 1-to-1 with the list of images.
        step = generation_kwargs["num_return_sequences"]
        results = []
        for i in range(0, len(texts), step):
            texts_chunk = texts[i : i + step]
            scores_chunk = scores[i : i + step]
            result = RecognitionResult(metadata, texts_chunk, scores_chunk)
            results.append(result)
        return results

    def _prepare_generation_kwargs(self, **kwargs):
        # Generally, all generation keyword arguments are passed to
        # the model's .generate method. However, to ensure that we
        # get the output format we want, some arguments needs to be
        # set (and potentially overridden).

        # Add default arguments
        kwargs = TrOCR.default_generation_kwargs | kwargs

        # HF defaults to greedy search if the user sets num_beams=1
        # But greedy search doesn't output sequence_scores, which we
        # we want to keep. Instead, we override num_beams to 2, and
        # set the return sequecens to 1.
        if kwargs.get("num_beams", None) == 1:
            kwargs["num_beams"] = 2
            kwargs["num_return_sequences"] = 1
        else:
            kwargs["num_return_sequences"] = kwargs["num_beams"]

        # Override arguments related to the output format
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        return kwargs
