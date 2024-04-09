from pathlib import Path
from typing import Optional

import numpy as np
from mmocr.apis import TextRecInferencer
from mmocr.structures import TextRecogDataSample

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.openmmlab.openmmlab_utils import SuppressOutput
from htrflow_core.models.torch_mixin import PytorchDeviceMixin
from htrflow_core.results import RecognizedText, Result


class Satrn(BaseModel, PytorchDeviceMixin):
    def __init__(
        self,
        model: str | Path = "model.pth",
        config: str | Path = "config.py",
        device: Optional[str] = None,
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
        *args,
    ) -> None:
        self.cache_dir = cache_dir
        with SuppressOutput():
            self.model = TextRecInferencer(
                model=config, weights=model, device=self.set_device(device), show_progress=False, *args
            )

        self.metadata = {"model": str(model)}

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        if len(images) > 1:
            batch_size = len(images)
        else:
            batch_size = 1

        outputs: TextRecogDataSample = self.model(
            images, batch_size=batch_size, draw_pred=False, return_datasample=True, **kwargs
        )

        return [self._create_text_result(image, output) for image, output in zip(images, outputs["predictions"])]

    def _create_text_result(self, image: np.ndarray, output: TextRecogDataSample) -> Result:
        results = []
        for i in range(0, len(texts), step):
            texts_chunk = texts[i : i + step]
            scores_chunk = scores[i : i + step]
            image_chunk = images[i // step]
            recognized_text = RecognizedText(texts=texts_chunk, scores=scores_chunk)
            result = Result.text_recognition_result(image=image_chunk, metadata=metadata, text=recognized_text)
            results.append(result)
        return results
