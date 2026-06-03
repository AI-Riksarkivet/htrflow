import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from htrflow.document import Region, Text
from htrflow.models.base_model import BaseModel
from htrflow.models.download import get_model_info
from htrflow.models.huggingface.mixins import ConfidenceMixin
from htrflow.utils.geometry import Bbox
from htrflow.utils.token_alignment import align_words_to_tokens


logger = logging.getLogger(__name__)


class TrOCR(BaseModel, ConfidenceMixin):
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
        model_kwargs = model_kwargs or {}
        self.decoding = model_kwargs.pop("decoding", None)
        self.model = VisionEncoderDecoderModel.from_pretrained(model, **model_kwargs)
        self.model.to(self.device)

        # Workaround for transformers' TrOCRSinusoidalPositionalEmbedding: its
        # get_embedding() always creates weights on CPU, and the checkpoint stores
        # them as a meta tensor (no actual data). Both issues cause a device mismatch
        # at inference time: position_ids are on CUDA but self.weights is on CPU/meta.
        # Fix: recompute the sinusoidal weights from scratch and place them on the
        # correct device. get_embedding() is pure math so no checkpoint data is needed.
        if hasattr(self.model.decoder, "model") and hasattr(self.model.decoder.model, "decoder"):
            embed_pos = getattr(self.model.decoder.model.decoder, "embed_positions", None)
            if embed_pos is not None and hasattr(embed_pos, "weights"):
                num_pos = embed_pos.weights.shape[0] if not embed_pos.weights.is_meta else 2048
                embed_pos.weights = embed_pos.get_embedding(
                    num_pos, embed_pos.embedding_dim, embed_pos.padding_idx
                ).to(self.device)

        logger.info("Initialized TrOCR model from %s on device %s.", model, self.model.device)

        # Initialize processor
        processor = processor or model
        processor_kwargs = processor_kwargs or {}
        self.processor = TrOCRProcessor.from_pretrained(processor, **processor_kwargs)
        logger.info("Initialized TrOCR processor from %s.", processor)

        self.metadata.update(
            {
                "model": model,
                "model_version": get_model_info(model, model_kwargs.get("revision", None)),
                "processor": processor,
                "processor_version": get_model_info(processor, processor_kwargs.get("revision", None)),
            }
        )

        # Map `compute_transition_scores` method from decoder to model
        self.compute_transition_scores = self.model.decoder.compute_transition_scores

    def _predict(self, images: list[Image], **generation_kwargs) -> list[Text]:
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
            scores = self.compute_sequence_confidence_score(model_outputs)

        # Assemble and return a list of Result objects from the prediction outputs.
        # `texts` and `scores` are flattened lists so we need to iterate over them in steps.
        # This is done to ensure that the list of results correspond 1-to-1 with the list of images.
        results = []
        step = generation_kwargs["num_return_sequences"]

        # Create the option to create a clean decoded text. Important for historical sources/models.
        # For example you can add generation_kwargs['decoding'] = 'unicode-escape'
        if self.decoding:
            for i, text in enumerate(texts):
                try:
                    texts[i] = text.encode("utf-8").decode(self.decoding)
                except UnicodeEncodeError as e:
                    logger.warning(
                        "Text %s could not be decoded with decoding '%s'. Error: %s", text, self.decoding, e
                    )
                    pass

        for i in range(0, len(texts), step):
            texts_chunk = texts[i : i + step]
            scores_chunk = scores[i : i + step]
            result = [Text(text=text, confidence=score) for text, score in zip(texts_chunk, scores_chunk)]
            results.append(result)
        return results


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

    def __init__(self, **kwargs):

        # We use the regular TrOCR but need to override the attention implementation.
        # Not all attention backends support `output_attentions` in model.generate,
        # which this model class bases its word segmentation on.
        model_kwargs = kwargs.get("model_kwargs", {})
        kwargs["model_kwargs"] = model_kwargs | {"attn_implementation": "eager"}
        super().__init__(**kwargs)

    def _predict(self, images: list[Image], **generation_kwargs) -> list[list[Text]]:
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
        n_tokens = outputs.sequences.shape[1] - 1  # -1 to ignore BOS token
        max_new_tokens = generation_kwargs.get("max_new_tokens")
        if max_new_tokens and n_tokens >= max_new_tokens:
            logger.warning(
                "The longest sequence of this batch has %d tokens, which is the"
                " maximum length, as specified by `max_new_tokens=%d`. This may"
                " indicate that the sequence was truncated.",
                n_tokens,
                max_new_tokens,
            )

        return self._postprocess(outputs, images)

    def _postprocess(self, outputs, images):

        heatmaps = self._create_attention_heatmaps(outputs)
        results = []
        for image, token_ids, heatmap in zip(images, outputs.sequences, heatmaps, strict=True):
            token_ids = token_ids[1:]
            word_spans = align_words_to_tokens(token_ids, self.processor.tokenizer)
            if not word_spans:
                results.append([])
                continue

            result = []
            for span in word_spans:
                word = self.processor.decode(token_ids[span.start : span.end], skip_special_tokens=True)
                if not word:
                    continue

                word_attention = heatmap[span.start : span.end].amax(axis=0)
                bbox = estimate_bbox_from_attention_heatmap(word_attention, image.size[::-1])

                if bbox is None:
                    # TODO
                    continue

                region = Region(
                    polygon=bbox.polygon(),
                    transcription=[Text(text=word)],
                )

                result.append(region)

            text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            line = Text(text=text)

            results.append([line] + result)
        return results

    def _create_attention_heatmaps(self, outputs) -> torch.Tensor:
        """
        Create attention heatmaps from the given outputs.

        Returns a tensor of size (N, S, H, W) where N is the batch size, S is
        the output sequence length and H and W are the height and width (in
        image patches) of the attention heatmap.
        """

        # outputs.cross_attentions is a list of attention scores with one entry
        # for each decoding step. We use the last step, which includes
        # information about the entire sequence.
        cross_attentions = torch.stack(outputs.cross_attentions[-1])

        # We want a single attention score for each pair of input tokens (image
        # patches) and output tokens, so we aggregate the 2nd and 0th
        # dimensions, which correspond to the attention heads and layers.
        attentions = cross_attentions.mean(axis=2).mean(axis=0)

        attentions = attentions * (attentions >= torch.amax(attentions, axis=1, keepdims=True))

        # Discard the CLS token attention
        attentions = attentions[:, :, 1:]

        # Unflatten the last dimension (image patches) into 2d
        image_height = self.processor.image_processor.size.height
        image_width = self.processor.image_processor.size.width
        patch_size = self.model.encoder.config.patch_size
        heatmap_height = image_height // patch_size
        heatmap_width = image_width // patch_size

        heatmaps = attentions.reshape(*attentions.shape[:2], heatmap_height, heatmap_width)
        return heatmaps


def estimate_bbox_from_attention_heatmap(
    heatmap: torch.Tensor,
    image_size: tuple[int, int],
    threshold: float = 0.25,
    ksize: tuple[int, int] = (15, 15),
) -> Bbox | None:
    """
    Estimate a bounding box around the "hottest" part of the given heatmap

    Arguments:
        heatmap: An attention heatmap.
        image_size: Size of input image as a (height, width) tuple.
        threshold: Parameter that controls the size of the bounding box. Should
            be within (0, 1). A higher threshold yields larger bounding boxes.
        ksize: Kernel size for gaussian smoothing. A larger kernel means larger
            bounding boxes.
    """

    # Step 1: Resize the attention heatmap
    height = 12  # We use a fixed small height for all heatmaps
    img_height, img_width = image_size
    img_aspect_ratio = img_height / img_width
    width = max(4, int(height / img_aspect_ratio))
    size = height, width
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size)
    heatmap = heatmap.squeeze(dim=0).squeeze(dim=0).cpu().numpy()

    # Smoothen the heatmap before estimating the bounding box
    smooth_heatmap = cv2.GaussianBlur(heatmap, ksize, 0)
    smooth_heatmap = smooth_heatmap / smooth_heatmap.max()

    # Estimate a bounding box around the main area
    _, thresh = cv2.threshold((smooth_heatmap * 255).astype(np.uint8), int(threshold * 255), 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Rescale bounding box to original coordinate system
    xmin = x * img_width / width
    ymin = y * img_height / height
    xmax = xmin + w * img_width / width
    ymax = ymin + h * img_height / height

    return Bbox(xmin, ymin, xmax, ymax)
