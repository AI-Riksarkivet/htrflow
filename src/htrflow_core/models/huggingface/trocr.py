import logging
from typing import Any

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation import BeamSearchEncoderDecoderOutput
from transformers.utils import ModelOutput

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import HF_CONFIG
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.results import Result


logger = logging.getLogger(__name__)


class TrOCR(BaseModel, PytorchMixin):
    """
    HTRFLOW adapter of the tranformer-based OCR model TrOCR.

    Uses huggingface's implementation of TrOCR. For further
    information, see
    https://huggingface.co/docs/transformers/model_doc/trocr.
    """

    def __init__(
        self,
        model: str,
        processor: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Initialize a TrOCR model

        Arguments:
            model: Path or name of pretrained VisisonEncoderDeocderModel.
            processor: Optional path or name of pretrained TrOCRProcessor.
                If not given, the model path or name is used.
            model_kwargs: Model initialization kwargs which are forwarded to
                VisionEncoderDecoderModel.from_pretrained.
            processor_kwargs: Processor initialization kwargs which are
                forwarded to TrOCRProcessor.from_pretrained.
            kwargs: Additional kwargs which are forwarded to BaseModel's
                __init__.
        """
        super().__init__(**kwargs)

        # Initialize model
        model_kwargs = HF_CONFIG | (model_kwargs or {})
        self.model = VisionEncoderDecoderModel.from_pretrained(model, **model_kwargs)
        self.model.to(self.set_device(self.device))
        logger.info("Initialized TrOCR model from %s on device %s.", model, self.model.device)

        # Initialize processor
        processor = processor or model
        processor_kwargs = HF_CONFIG | (processor_kwargs or {})
        self.processor = TrOCRProcessor.from_pretrained(processor, **processor_kwargs)
        logger.info("Initialized TrOCR processor from %s.", processor)

        self.metadata.update(
            {
                "model": str(model),
                "processor": str(processor),
                "framework": Framework.HuggingFace.value,
                "task": Task.Image2Text.value,
                "device": self.device_id,
            }
        )

    def _predict(self, images: list[np.ndarray], **generation_kwargs) -> list[Result]:
        """Perform inference on `images`

        Arguments:
            images: Input images.
            **generation_kwargs: Optional keyword arguments that are
                forwarded to the model's .generate() method.

        Returns:
            The predicted texts and confidence scores as a list of `Result` instances.
        """

        # Prepare generation keyword arguments: Generally, all kwargs are
        # forwarded to the model's .generate method, but some need to be
        # explicitly set (and possibly overridden) to ensure that we get the
        # output format we want.
        generation_kwargs["num_return_sequences"] = generation_kwargs.get("num_beams", 1)
        generation_kwargs["output_scores"] = True
        generation_kwargs["return_dict_in_generate"] = True

        # Do inference
        model_inputs = self.processor(images, return_tensors="pt").pixel_values
        model_outputs = self.model.generate(model_inputs.to(self.model.device), **generation_kwargs)

        texts = self.processor.batch_decode(model_outputs.sequences, skip_special_tokens=True)
        scores = self._compute_seuqence_scores(model_outputs)

        # Assemble and return a list of Result objects from the prediction outputs.
        # `texts` and `scores` are flattened lists so we need to iterate over them in steps.
        # This is done to ensure that the list of results correspond 1-to-1 with the list of images.
        results = []
        metadata = self.metadata | {"generation_kwargs": generation_kwargs}
        step = generation_kwargs["num_return_sequences"]
        for i in range(0, len(texts), step):
            texts_chunk = texts[i : i + step]
            scores_chunk = scores[i : i + step]
            result = Result.text_recognition_result(metadata, texts_chunk, scores_chunk)
            results.append(result)
        return results

    def _compute_seuqence_scores(self, outputs: ModelOutput):
        """Compute normalized prediction score for each output sequence

        This function computes the normalized sequence scores from the output.
        (Contrary to sequence_scores, which returns unnormalized scores)
        It follows example #1 found here:
        https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
        """

        if isinstance(outputs, BeamSearchEncoderDecoderOutput):
            transition_scores = self.model.decoder.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
            )
        else:
            transition_scores = self.model.decoder.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
        transition_scores = transition_scores.cpu()
        length_penalty = self.model.generation_config.length_penalty
        output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        return np.exp(scores).tolist()
