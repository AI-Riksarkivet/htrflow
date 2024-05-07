import logging

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation import BeamSearchEncoderDecoderOutput
from transformers.utils import ModelOutput

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import HF_CONFIG
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.results import RecognizedText, Result


logger = logging.getLogger(__name__)


class TrOCR(BaseModel, PytorchMixin):
    default_generation_kwargs = {
        "num_beams": 4,
    }

    def __init__(
        self,
        model: str = "microsoft/trocr-base-handwritten",
        processor: str = "microsoft/trocr-base-handwritten",
        *model_args,
        **kwargs,
    ):
        """Initialize a TrOCR model

        Arguments:
            model: Path or name of pretrained VisisonEncoderDeocderModel.
                Defaults to 'microsoft/trocr-base-handwritten'.
            processor: Path or name of pretrained TrOCRProcessor.
                Defaults to 'microsoft/trocr-base-handwritten'.
        """
        super().__init__(**kwargs)

        self.model = VisionEncoderDecoderModel.from_pretrained(model, *model_args, **HF_CONFIG)
        self.model.to(self.set_device(self.device))
        logger.info("Initialized TrOCR model from %s on device %s", model, self.model.device)

        processor = processor or model

        self.processor = TrOCRProcessor.from_pretrained(processor, **HF_CONFIG)
        logger.info("Initialized TrOCR processor from %s", processor)

        self.metadata.update(
            {
                "model": str(model),
                "processor": str(processor),
                "framework": Framework.HuggingFace.value,
                "task": Task.Image2Text.value,
                "device": self.device_id,
            }
        )

    def filter_text_on_thresh():
        # TODO: Should add **kwargs: that also filter output.
        # For instance pred_threshold = 0.7 should filter conf score.
        pass

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
        metadata = self.metadata | ({"generation_args": generation_kwargs} if generation_kwargs else {})

        model_inputs = self.processor(images, return_tensors="pt").pixel_values
        model_outputs = self.model.generate(model_inputs.to(self.model.device), **generation_kwargs)

        texts = self.processor.batch_decode(model_outputs.sequences, skip_special_tokens=True)

        scores = self.compute_seuqence_scores(model_outputs)
        step = generation_kwargs["num_return_sequences"]

        return self._create_text_results(images, texts, scores, metadata, step)

    def _create_text_results(
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
        """
        # Add default arguments
        kwargs = TrOCR.default_generation_kwargs | kwargs
        kwargs["num_return_sequences"] = kwargs.get("num_beams", 1)

        # Override arguments related to the output format
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        return kwargs

    def compute_seuqence_scores(self, outputs: ModelOutput):
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
