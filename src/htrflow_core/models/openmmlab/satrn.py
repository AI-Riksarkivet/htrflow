from os import PathLike
from typing import Optional

import numpy as np
from mmocr.apis import TextRecInferencer
from mmocr.structures import TextRecogDataSample

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.openmmlab import openmmlab_downloader
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.models.torch_mixin import PytorchDeviceMixin
from htrflow_core.results import RecognizedText, Result


class Satrn(BaseModel, PytorchDeviceMixin):
    def __init__(
        self,
        model: str | PathLike = "Riksarkivet/satrn_htr",
        config: str | PathLike = "Riksarkivet/satrn_htr",
        device: Optional[str] = None,
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
        *args,
    ) -> None:
        self.cache_dir = cache_dir

        model_weights, model_config = openmmlab_downloader.load_from_hf(model, config, cache_dir, hf_token)

        with SuppressOutput():
            self.model = TextRecInferencer(
                model=model_config, weights=model_weights, device=self.set_device(device), *args
            )

        self.metadata = {"model": str(model), "config": str(config)}

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
        recognized_text = RecognizedText(texts=output["text"], scores=output["scores"])
        return Result.text_recognition_result(image=image, metadata=self.metadata, text=recognized_text)


if __name__ == "__main__":
    img = "/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/data/demo_images/image_norhand2.png"

    model = Satrn(model="Riksarkivet/satrn_htr")

    results = model([img] * 2, batch_size=2)

    print(results[0].data)
