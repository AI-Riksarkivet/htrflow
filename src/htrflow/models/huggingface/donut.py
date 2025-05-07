import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np
from transformers import DonutProcessor, VisionEncoderDecoderModel

from htrflow.models.base_model import BaseModel
from htrflow.models.download import get_model_info
from htrflow.models.huggingface.mixins import ConfidenceMixin
from htrflow.results import Result


logger = logging.getLogger(__name__)


class Donut(BaseModel, ConfidenceMixin):
    """
    HTRflow adapter of Donut model.
    """

    def __init__(self,
        model: str,
        processor: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        prompt: str = "<s>",
        **kwargs,
    ):
        """
        Arguments:
            model: Path or name of pretrained VisisonEncoderDeocderModel.
            processor: Optional path or name of pretrained DonutProcessor. If not given, the model path or name is
                used.
            model_kwargs: Model initialization kwargs which are forwarded to
                VisionEncoderDecoderModel.from_pretrained.
            processor_kwargs: Processor initialization kwargs which are forwarded to DonutProcessor.from_pretrained.
            prompt: Task prompt to use for all decoding tasks. Defaults to '<s>'.
            kwargs: Additional kwargs which are forwarded to BaseModel's __init__.
        """
        super().__init__(**kwargs)

        # Initialize model
        model_kwargs = model_kwargs or {}
        self.model = VisionEncoderDecoderModel.from_pretrained(model, **model_kwargs)
        self.model.to(self.device)
        logger.info("Initialized Donut model from %s on device %s.", model, self.model.device)

        # Initialize processor
        processor = processor or model
        processor_kwargs = processor_kwargs or {}
        self.processor = DonutProcessor.from_pretrained(processor, **processor_kwargs)
        logger.info("Initialized Donut processor from %s.", processor)

        self.prompt = prompt

        self.metadata["model"] = model
        self.metadata["model_version"] = get_model_info(model, model_kwargs.get("revision", None))
        self.metadata["processor"] = processor
        self.metadata["processor_version"]= get_model_info(processor, processor_kwargs.get("revision", None))

        self.compute_transition_scores = self.model.decoder.compute_transition_scores

    def _predict(self, images: list[np.ndarray], **generation_kwargs) -> list[Result]:

        # Prepare generation kwargs
        defaults = {
            "max_length": self.model.decoder.config.max_position_embeddings,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "bad_words_ids": [[self.processor.tokenizer.unk_token_id]],
        }
        overrides = {
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        generation_kwargs = defaults | generation_kwargs | overrides
        warn_when_overridden(generation_kwargs, overrides)

        # Run inference
        prompts = [self.prompt for _ in images]
        inputs = self.processor(images, prompts, return_tensors="pt")
        outputs = self.model.generate(
            inputs.pixel_values.to(self.model.device),
            decoder_input_ids=inputs.input_ids.to(self.model.device),
            **generation_kwargs
        )
        scores = self.compute_sequence_confidence_score(outputs)
        token_scores = self.compute_confidence_per_token(outputs)

        # Construct results
        results = []
        for sequence, score, token_scores in zip(self.processor.batch_decode(outputs.sequences), scores, token_scores):
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
            sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
            data = self.processor.token2json(sequence)
            results.append(data | {"sequence_confidence_score": score, "token_confidence_scores": token_scores})

        chunked_results = []
        chunk_size = generation_kwargs.get("num_return_sequences", 1)
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            chunked_results.append(Result(data={"donut_result": chunk}))

        return chunked_results


def warn_when_overridden(kwargs: dict, overrides: dict):
    """
    Log a warning if any of the given keyword arguments are overridden.

    Arguments:
        kwargs: Given keyword arguments.
        overrides: Keyword argument overrides.
    """
    for key, value in kwargs.items():
        if key in overrides:
            if overrides[key] != value:
                msg = "HTRflow Donut model does not support '%s'='%s'. Using '%s'='%s' instead."
                _warn_once(msg, key, value, overrides[key])

@lru_cache
def _warn_once(msg, *args):
    """Log `msg` once"""
    logger.warning(msg, *args)
