import logging

import numpy as np
from mmocr.apis import TextRecInferencer
from mmocr.structures import TextRecogDataSample

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import MMLabsDownloader
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.results import RecognizedText, Result


logger = logging.getLogger(__name__)


class Satrn(BaseModel, PytorchMixin):
    def __init__(
        self,
        model: str = "Riksarkivet/satrn_htr",
        config: str = "Riksarkivet/satrn_htr",
        *model_args,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        model_weights, model_config = MMLabsDownloader.from_pretrained(model, config, self.cache_dir)

        with SuppressOutput():
            self.model = TextRecInferencer(
                model=model_config, weights=model_weights, device=self.set_device(self.device), *model_args
            )

        logger.info(f"Model loaded on ({self.device_id}) from {model}.")

        self.metadata.update(
            {
                "model": str(model),
                "config": str(config),
                "framework": Framework.Openmmlab.value,
                "task": Task.Image2Text.value,
                "device": self.device_id,
            }
        )

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        if len(images) > 1:
            batch_size = len(images)
        else:
            batch_size = 1

        outputs: TextRecogDataSample = self.model(
            images, batch_size=batch_size, return_datasamples=False, progress_bar=False, **kwargs
        )

        return [self._create_text_result(image, output) for image, output in zip(images, outputs["predictions"])]

    def _create_text_result(self, image: np.ndarray, output: list) -> Result:
        recognized_text = RecognizedText(texts=[output["text"]], scores=[output["scores"]])
        return Result.text_recognition_result(image=image, metadata=self.metadata, text=recognized_text)
