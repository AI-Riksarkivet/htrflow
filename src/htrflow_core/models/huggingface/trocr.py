from pathlib import Path
from typing import Optional

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.torch_mixin import PytorchDeviceMixin
from htrflow_core.results import RecognizedText, Result


class TrOCR(BaseModel, PytorchDeviceMixin):
    default_generation_kwargs = {
        "num_beams": 4,
    }

    def __init__(
        self,
        model: str | Path = "microsoft/trocr-base-handwritten",
        processor: str = "microsoft/trocr-base-handwritten",
        device: Optional[str] = None,
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
    ):
        """Initialize a TrOCR model

        Arguments:
            model: Path or name of pretrained VisisonEncoderDeocderModel.
                Defaults to 'microsoft/trocr-base-handwritten'.
            processor: Path or name of pretrained TrOCRProcessor.
                Defaults to 'microsoft/trocr-base-handwritten'.
        """

        self.cache_dir = cache_dir
        self.model = VisionEncoderDecoderModel.from_pretrained(model, cache_dir=cache_dir, token=hf_token).to(
            self.set_device(device)
        )

        if processor is None:
            processor = model
        self.processor = TrOCRProcessor.from_pretrained(processor, cache_dir=cache_dir, token=hf_token)

        self.metadata = {
            "model": str(model),
            "processor": str(processor),
        }

    def _predict(self, images: list[np.ndarray], **generation_kwargs) -> list[Result]:
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

        model_inputs = self.processor(images, return_tensors="pt").pixel_values
        model_outputs = self.model.generate(model_inputs.to(self.model.device), **generation_kwargs)

        texts = self.processor.batch_decode(model_outputs.sequences, skip_special_tokens=True)

        scores = model_outputs.sequences_scores.tolist()
        step = generation_kwargs["num_return_sequences"]

        return self._create_text_result(images, texts, scores, metadata, step)

    def _create_text_result(
        self, images: list[np.ndarray], texts: list[str], scores: list[float], metadata: dict, step: int
    ) -> list[Result]:
        """Assemble and return a list of Result objects from the prediction outputs.
        `texts` and `scores` are flattened lists so we need to iterate over them in steps.
        This is done to ensure that the list of results correspond 1-to-1 with the list of images.
        """
        results = []
        for i in range(0, len(texts), step):
            texts_chunk = texts[i : i + step]
            scores_chunk = scores[i : i + step]
            image_chunk = images[i // step]
            recognized_text = RecognizedText(texts=texts_chunk, scores=scores_chunk)
            result = Result.text_recognition_result(image=image_chunk, metadata=metadata, text=recognized_text)
            results.append(result)
        return results

    def _prepare_generation_kwargs(self, **kwargs):
        """
        Generally, all generation keyword arguments are passed to
        the model's .generate method. However, to ensure that we
        get the output format we want, some arguments needs to be
        set (and potentially overridden).

        HF defaults to greedy search if the user sets num_beams=1
        But greedy search doesn't output sequence_scores, which we
        we want to keep. Instead, we override num_beams to 2, and
        set the return sequecens to 1.

        """

        # Add default arguments
        kwargs = TrOCR.default_generation_kwargs | kwargs

        if kwargs.get("num_beams", None) == 1:
            kwargs["num_beams"] = 2
            kwargs["num_return_sequences"] = 1
        else:
            kwargs["num_return_sequences"] = kwargs["num_beams"]

        # Override arguments related to the output format
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        return kwargs
