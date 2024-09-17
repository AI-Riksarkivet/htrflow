import logging
from typing import Any

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import ModelOutput

from htrflow.models.base_model import BaseModel
from htrflow.models.hf_utils import HF_CONFIG
from htrflow.results import Result


logger = logging.getLogger(__name__)


class TrOCR(BaseModel):
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
        """
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
        self.model.to(self.device)
        logger.info("Initialized TrOCR model from %s on device %s.", model, self.model.device)

        # Initialize processor
        processor = processor or model
        processor_kwargs = HF_CONFIG | (processor_kwargs or {})
        self.processor = TrOCRProcessor.from_pretrained(processor, **processor_kwargs)
        logger.info("Initialized TrOCR processor from %s.", processor)

        self.metadata.update(
            {
                "model": model,
                "model_version": model_info(model).sha,
                "processor": processor,
                "processor_version": model_info(processor).sha,
            }
        )

    def _predict(self, images: list[np.ndarray], **generation_kwargs) -> list[Result]:
        """TrOCR-specific prediction method.

        This method is used by `predict()` and should typically not be
        called directly.

        Arguments:
            images: Input images.
            **generation_kwargs: Optional keyword arguments that are
                forwarded to the model's .generate() method.
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
        scores = self._compute_sequence_scores(model_outputs)

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

    def _compute_sequence_scores(self, outputs: ModelOutput):
        """Compute normalized prediction score for each output sequence

        This function computes the normalized sequence scores from the output.
        (Contrary to sequence_scores, which returns unnormalized scores)
        It follows example #1 found here:
        https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
        """
        transition_scores = self.model.decoder.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            beam_indices=getattr(outputs, "beam_indices", None),
            normalize_logits=True,
        )

        length_penalty = self.model.generation_config.length_penalty
        # In output from greedy decoding, padding tokens have a transition score
        # of negative infinity. To "hide" them from the score computation
        # they are set to 0 instead.
        transition_scores[outputs.sequences[:, 1:] == self.model.generation_config.pad_token_id] = 0
        transition_scores = transition_scores.cpu()
        output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        return np.exp(scores).tolist()


class WordLevelTrOCR(TrOCR):
    """A version of TrOCR which outputs words instead of lines.

    This TrOCR wrapper uses the model's attention weights to estimate
    word boundaries. See notebook [TODO: link] for more details. It
    does not support beam search, but can otherwise be used as a drop-
    in replacement of TrOCR.

    Example usage with the `TextRecognition` pipeline step:
    ```yaml
    - step: TextRecognition
      settings:
        model: WordLevelTrOCR
        model_kwargs:
          ...
        processor: ...
        processor_kwargs:
          ...
        generation_settings:
          batch_size: 32
    ```
    """

    def _predict(self, images: list[np.ndarray], **generation_kwargs) -> list[Result]:
        config_overrides = {
            "output_scores": True,
            "output_attentions": True,
            "return_dict_in_generate": True,
            "num_beams": 1,
            "early_stopping": False,
            "length_penalty": None,
        }

        for key, value in config_overrides.items():
            if key in generation_kwargs and generation_kwargs[key] != value:
                logger.warning(
                    "WordLevelTrOCR does not support %s=%s. Using %s=%s instead.",
                    key,
                    generation_kwargs[key],
                    key,
                    value,
                )

        inputs = self.processor(images, return_tensors="pt").pixel_values
        outputs = self.model.generate(
            inputs.to(self.model.device),
            **(generation_kwargs | config_overrides),
        )

        # Get the attention weights at the last decoding step
        attentions = torch.stack(outputs.cross_attentions[-1])
        n_tokens = attentions.shape[3]
        # Aggregate all attention weights for each token. Here, we use the
        # mean of the attention heads at each layer (axis 2) to get one set of
        # weights per token and layer, and then summing over the layers (axis 0)
        # to get one set of weights per token.
        attentions = attentions.mean(axis=2).sum(axis=0)

        # Create heatmaps by reshaping the weights dimension (size n_patches * n_patches + 1)
        # to (n_patches, n_patches) and discard the extra first patch.
        encoder_config = self.model.config.encoder
        n_patches = int(encoder_config.image_size / encoder_config.encoder_stride)
        heatmaps = torch.reshape(attentions[:, :, 1:], (-1, n_tokens, n_patches, n_patches))

        lines = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        line_scores = self._compute_sequence_scores(outputs)
        special_tokens = {*self.processor.tokenizer.special_tokens_map.values()}
        results = []

        for i, sequence in enumerate(outputs.sequences):
            tokens = self.processor.batch_decode(sequence)

            # Deriving the words from the line (and not by joining the tokens) is a
            # work-around in order to decode special characters correctly.
            words = [word if len(word) else " " for word in lines[i].split(" ")]

            height, width = images[i].shape[:2]
            spaces = attention_based_wordseg(tokens, heatmaps[i], special_tokens, width)
            word_boundaries = list(zip(spaces, spaces[1:]))

            if any(start >= end_ for start, end_ in word_boundaries):
                word_boundaries = [(0, width) for _ in words]
                logger.warning("Word segmentation failed on line with detected text: %s", lines[i])

            results.append(
                Result.word_segmentation_result(
                    metadata=self.metadata,
                    orig_shape=(height, width),
                    words=words,
                    word_scores=[line_scores[i] for word in words],
                    line=lines[i],
                    line_score=line_scores[i],
                    bboxes=[(start, 0, end_, height) for start, end_ in word_boundaries],
                )
            )
        return results


def attention_based_wordseg(tokens, heatmaps, skip_tokens=None, full_width=1):
    tokens = tokens[1:]
    n_tokens = len(tokens)
    heatmaps = heatmaps[:n_tokens]
    if skip_tokens is None:
        skip_tokens = []

    # Subtract the mean from the heatmaps.
    heatmaps -= heatmaps.mean(axis=0)
    heatmaps[heatmaps < 0] = 0

    # Token indices of spaces / word boundaries
    spaces = [i for i, token in enumerate(tokens) if token.startswith(" ") and len(token) > 1]

    word_heatmaps = []
    for word_start, word_end in zip([0] + spaces, spaces + [n_tokens]):
        token_weights = [0 if token in skip_tokens else len(token) for token in tokens[word_start:word_end]]
        token_weights = torch.tensor(token_weights).to(heatmaps.device)
        token_heatmaps = heatmaps[word_start:word_end]
        token_heatmaps = token_heatmaps.permute(*torch.arange(token_heatmaps.ndim - 1, -1, -1))
        word_heatmap = torch.mul(token_heatmaps, token_weights).sum(axis=1)
        word_heatmaps.append(word_heatmap)

    columns = [word_heatmap.sum(axis=1) for word_heatmap in word_heatmaps]
    columns = torch.stack(columns)
    columns = torch.div(columns.T, columns.max(axis=1).values).T

    intersections = [0] + _find_intersections(columns) + [1]
    return [x * full_width for x in intersections]


def _find_intersections(columns):
    argmaxs = columns.argmax(axis=1)
    result = []
    for i, lo in enumerate(argmaxs[:-1]):
        cols1 = columns[i]
        cols2 = columns[i + 1]

        hi = argmaxs[i + 1] + 1
        intersections = torch.argwhere(cols1[lo:hi] <= cols2[lo:hi])
        if len(intersections):
            intersection = intersections[0][0]
        else:
            intersection = int((lo + hi) / 2)
        result.append(int(intersection + lo))
    return [x / columns.shape[1] for x in result]
