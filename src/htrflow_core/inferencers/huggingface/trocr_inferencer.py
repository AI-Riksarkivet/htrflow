import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation import BeamSearchEncoderDecoderOutput
from transformers.utils import ModelOutput
from htrflow_core.inferencers.base_inferencer import BaseInferencer


class TrOCRInferencer(BaseInferencer):

    default_config = {
        "num_beams": 4,
    }

    def __init__(self, model: VisionEncoderDecoderModel, processor: TrOCRProcessor):
        self._model = model
        self._processor = processor

    def preprocess(self, images: list[str]):
        images = [Image.open(image).convert("RGB") for image in images]
        return self._processor(images, return_tensors="pt").pixel_values

    def predict(self, images: list[str], **kwargs):
        """Run inference on `images`

        Uses beam search with 4 beams by default. This can be altered by passing `num_beams` in `kwargs`.

        Args:
            images (list[str]): List of image paths
            kwargs: Keyword arguments passed to model.generate()

        Returns:
            A tuple (texts, scores)
        """

        # Add default arguments
        kwargs = TrOCRInferencer.default_config | kwargs

        # `num_return_sequences` defaults to `num_beams`
        kwargs["num_return_sequences"] = kwargs.get("num_return_sequences", kwargs["num_beams"])

        # Overwrite mandatory arguments
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True

        inputs = self.preprocess(images)
        outputs = self._model.generate(inputs, **kwargs)
        return self.postprocess(outputs, kwargs["num_return_sequences"])

    def postprocess(self, outputs: BeamSearchEncoderDecoderOutput | ModelOutput, num_return_sequences: int):
        texts = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        scores = self._sequence_scores(outputs)

        # Beam search returns all sequences in a flattened list so we need to re-group them so that each item in the
        # final output list correspond to one input line
        if num_return_sequences > 1:
            texts = _group(texts, num_return_sequences)
            scores = _group(self._sequence_scores(outputs), num_return_sequences)

        return texts, scores

    def _sequence_scores(self, outputs: ModelOutput) -> list[float] | list[list[float]]:
        """Compute prediction score for each output sequence"""

        # This implementation follows example #1 found here:
        # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
        # Note that outputs.sequences_scores aren't used - those are based on unnormalized logits. This implementation
        # normalizes the logits before computing the sequences' prediction scores, which should make the prediction
        # scores produced by beam search and greedy decoding comparable with each other.

        if isinstance(outputs, BeamSearchEncoderDecoderOutput):
            transition_scores = self._model.decoder.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
            )
        else:
            transition_scores = self._model.decoder.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

        length_penalty = self._model.generation_config.length_penalty
        output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        return scores.tolist()


def _group(lst: list, n: int) -> list:
    """Group items in `lst` in sublists of length `n`"""
    return [lst[i : i + n] for i in range(0, len(lst), n)]