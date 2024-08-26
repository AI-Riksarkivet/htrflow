import logging
import re
import time
from threading import Thread
from typing import Optional, Union

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TextIteratorStreamer,
    TextStreamer,
)

from htrflow.models.base_model import BaseModel
from htrflow.results import RecognizedText, Result
from htrflow.utils import imgproc


logger = logging.getLogger(__name__)


class LLavaNext(BaseModel):
    default_generation_kwargs = {"num_beams": 1, "max_new_tokens": 200}

    def __init__(
        self,
        model: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        processor: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        prompt: str = "[INST] <image>\Please transcribe the handwritten English text displayed in the image [/INST]",
        device: str | None = None,
        *model_args,
    ):
        super().__init__(device)

        # nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model,
            cache_dir=self.cache_dir,
            token=True,
            # quantization_config=nf4_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            *model_args,
        )

        self.model.to(self.device)

        processor = processor or model

        self.processor = LlavaNextProcessor.from_pretrained(processor, cache_dir=self.cache_dir, token=True)

        self.prompt = prompt

        logger.info(f"Model loaded on ({self.device_id}) from {model}.")

        logger.info(f"Processor loaded from {processor}.")

        self.metadata.update(
            {
                "model": model,
                "model_version": model_info(model).sha,
                "prompt": prompt,
                "processor": processor,
                "processor_version": model_info(processor).sha,
            }
        )

    def _predict(self, images: list[np.ndarray], **generation_kwargs) -> list[Result]:
        generation_kwargs = self._prepare_generation_kwargs(**generation_kwargs)
        metadata = self.metadata | ({"generation_args": generation_kwargs} if generation_kwargs else {})

        prompts = [self.prompt] * len(images)

        model_inputs = self.processor(prompts, images=images, return_tensors="pt", padding=True).to(self.model.device)
        model_outputs = self.model.generate(**model_inputs, **generation_kwargs)

        texts = self.processor.batch_decode(model_outputs.sequences, skip_special_tokens=True)

        scores = model_outputs.sequences_scores.tolist()
        step = generation_kwargs["num_return_sequences"]

        return self._create_text_results(images, texts, scores, metadata, step)

    def stream_predict(
        self,
        image: Union[np.ndarray, str],
        prompt: Optional[str] = None,
        iterator: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Predicts output from the model using streaming to handle the input and output progressively.

        Args:
            image (np.ndarray): The image array.
            prompt (Optional[str]): An optional prompt to use alongside the image.
            iterator (bool): Whether to use an iterator-based approach to streaming.
            **generation_kwargs: Additional keyword arguments for generation.

        Returns:
            str: The generated text.
        """

        img = imgproc.read(image)

        inputs = self.processor(prompt, img, return_tensors="pt").to(self.model.device)

        if iterator:
            return self._handle_iterator_stream(inputs, **generation_kwargs)
        else:
            return self._handle_standard_stream(inputs, **generation_kwargs)

    def _handle_iterator_stream(self, inputs, **generation_kwargs):
        streamer = TextIteratorStreamer(self.processor, skip_prompt=True)

        new_generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        new_generation_kwargs.update(**generation_kwargs)

        thread = Thread(target=self.model.generate, kwargs=new_generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            time.sleep(0.04)
            yield buffer

    def _handle_standard_stream(self, inputs, **generation_kwargs) -> str:
        streamer = TextStreamer(self.processor, skip_prompt=True)
        output = self.model.generate(**inputs, streamer=streamer, **generation_kwargs)
        return self.processor.decode(output[0], skip_special_tokens=True)

    def _create_text_results(
        self,
        images: list[np.ndarray],
        texts: list[str],
        scores: list[float],
        metadata: dict,
        step: int,
    ) -> list[Result]:
        results = []
        for i in range(0, len(texts), step):
            texts_chunk = texts[i : i + step]
            scores_chunk = scores[i : i + step]

            without_prompt_texts_chunk = [re.sub(r"\[INST\].*?\[/INST\]", "", text) for text in texts_chunk]

            recognized_text = RecognizedText(texts=without_prompt_texts_chunk, scores=scores_chunk)
            result = Result.text_recognition_result(metadata=metadata, text=recognized_text)
            results.append(result)
        return results

    def _prepare_generation_kwargs(self, **kwargs):
        kwargs = LLavaNext.default_generation_kwargs | kwargs

        if kwargs.get("num_beams", None) == 1:
            kwargs["num_beams"] = 2
            kwargs["num_return_sequences"] = 1
        else:
            kwargs["num_return_sequences"] = kwargs["num_beams"]

        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        return kwargs
