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
    HTRflow adapter of the tranformer-based OCR model TrOCR.

    Uses huggingface's implementation of TrOCR. For further
    information, see
    https://huggingface.co/docs/transformers/model_doc/trocr.

    Example usage with the `TextRecognition` step:
    ```yaml
    - step: TextRecognition
      settings:
        model: TrOCR
        model_settings:
          model: Riksarkivet/trocr-base-handwritten-hist-swe-2
          device: cpu
          model_kwargs:
            revision: 6ecbb5d643430385e1557001ae78682936f8747f
        generation_settings:
          batch_size: 8
          num_beams: 1
    ```
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
        called directly. However, `predict()` forwards additional kwargs
        to this method.

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
        with torch.no_grad():
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
        beam_indices = getattr(outputs, "beam_indices", None)
        transition_scores = self.model.decoder.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )

        is_beam_search = beam_indices is not None
        length_penalty = self.model.generation_config.length_penalty if is_beam_search else 1.0
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

    Example usage with the `TextRecognition` step:
    ```yaml
    - step: TextRecognition
      settings:
        model: WordLevelTrOCR
        model_settings:
          model: Riksarkivet/trocr-base-handwritten-hist-swe-2
          device: cpu
          model_kwargs:
            revision: 6ecbb5d643430385e1557001ae78682936f8747f
        generation_settings:
          batch_size: 8
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

        # Warn if `max_new_tokens` was given and the limit was reached
        n_tokens = outputs.sequences.shape[1]
        max_new_tokens = generation_kwargs.get("max_new_tokens")
        if max_new_tokens and n_tokens >= max_new_tokens + 1:  # +1 to account for the BOS token
            logger.warning(
                "The longest sequence of this batch has %d tokens, which is the"
                " maximum length, as specified by `max_new_tokens=%d`. This may"
                " indicate that the sequence was truncated.",
                n_tokens,
                max_new_tokens,
            )

        attentions = aggregate_attentions(outputs.cross_attentions)

        # Create heatmaps by reshaping the weights dimension (size n_patches * n_patches + 1)
        # to (n_patches, n_patches) and discard the extra first patch.
        encoder_config = self.model.config.encoder
        n_patches = int(encoder_config.image_size / encoder_config.encoder_stride)
        n_tokens = len(outputs.cross_attentions)
        heatmaps = torch.reshape(attentions[:, :, 1:], (n_tokens, -1, n_patches, n_patches))

        lines = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True, cleanup_tokenization_spaces=False
        )
        line_scores = self._compute_sequence_scores(outputs)
        special_tokens = {*self.processor.tokenizer.special_tokens_map.values()}
        results = []

        for i, sequence in enumerate(outputs.sequences):
            tokens = self.processor.batch_decode(sequence)

            # Deriving the words from the line (and not by joining the tokens) is a
            # work-around in order to decode special characters correctly.
            words = [word if len(word) else " " for word in lines[i].split(" ")]

            height, width = images[i].shape[:2]
            spaces = attention_based_wordseg(tokens, heatmaps[:, i, :, :], special_tokens, width)
            word_boundaries = list(zip(spaces, spaces[1:]))

            if len(word_boundaries) != len(words) or any(start >= end_ for start, end_ in word_boundaries):
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


def attention_based_wordseg(
    tokens: list[str], heatmaps: torch.Tensor, skip_tokens: list[str] | None = None, image_width: int = 1
) -> list[float]:
    """
    Estimate word segmentation based on attention scores

    Arguments:
        tokens: list of N tokens
        heatmaps: tensor of attention weights for each token, shape (N, img_height, img_width)
        skip_tokens: list of tokens to not include in word attention scores
        image_width: the width of the original image

    Returns:
        An estimated word segmenation as a list of x-coordinates of the estimated spaces,
        including 0 and `image_width`.
    """

    tokens = tokens[1:]  # Skip the first special token
    n_tokens = len(tokens)
    heatmaps = heatmaps[:n_tokens]
    if skip_tokens is None:
        skip_tokens = []

    # Sum along the columns to get attention scores along the x axis
    xaxis_attention_scores = heatmaps.sum(axis=1)  #  (N, img_width)

    # Denoise attention scores by only keeping the high attention scores
    xaxis_attention_scores -= xaxis_attention_scores.mean(axis=0)
    xaxis_attention_scores[xaxis_attention_scores < 0] = 0

    # Token indices of spaces / word boundaries
    spaces = [i for i, token in enumerate(tokens) if token.startswith(" ") and len(token) > 1]

    # Create word attention scores by doing a weighted sum over the token attention scores
    word_attention_scores = []
    for word_start, word_end in zip([0] + spaces, spaces + [n_tokens]):
        token_weights = [0 if token in skip_tokens else len(token) for token in tokens[word_start:word_end]]
        token_weights = torch.tensor(token_weights).reshape(-1, 1).to(heatmaps.device)
        word_attention_scores.append((token_weights * xaxis_attention_scores[word_start:word_end]).sum(axis=0))
    word_attention_scores = torch.stack(word_attention_scores)  # (n_words, img_width)

    # Normalize word attention scores such that each score is in (0, 1]
    normalized_word_attention_scores = torch.div(word_attention_scores.T, word_attention_scores.max(axis=1).values).T

    intersections = [0] + _find_intersections(normalized_word_attention_scores) + [1]
    return [x * image_width for x in intersections]


def aggregate_attentions(cross_attention: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
    """
    Combine cross_attention output to one attention tensor per token

    The `cross_attentions` outputted by `model.generate(output_attentions=True)`
    is a tuple (one per each token) of tuples (one per each layer) of attention
    matrices (tensors of shape (batch_size, num_heads, generated_length,
    input_sequence_length)).

    This function aggregates all this information into a tensor of shape
    (num_tokens, input_sequence_length), by for each token:
        1. Summing the attention weights of all layers
        2. Taking the mean attention weights of all attention heads
    This method of aggregation is a result of quite limited trial and error -
    it seems to work fine for the task, but there are most certainly other,
    possibly better, methods.

    Args:
        cross_attention: Cross attention weights as outputted by
            `model.generate(output_attentions=True)`

    Returns:
        A tensor of shape (num_tokens, input_sequence_length) with aggregated
        attention weights for each token.
    """
    aggregated = []
    for token_attention in cross_attention:
        aggregated.append(torch.stack(token_attention).mean(axis=2).sum(axis=0)[:, -1, :])
    return torch.stack(aggregated)


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
