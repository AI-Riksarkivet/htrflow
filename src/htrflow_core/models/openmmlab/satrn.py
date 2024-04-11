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
            self.model = TextRecInferencer(model=config, weights=model, device=self.set_device(device))

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
        print(output)
        recognized_text = RecognizedText(texts=output["text"], scores=output["scores"])
        return Result.text_recognition_result(image=image, metadata=self.metadata, text=recognized_text)


if __name__ == "__main__":
    img = "/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/data/demo_images/trocr_demo_image.png"

    # from huggingface_hub import hf_hub_download

    # model_path = hf_hub_download(
    #     repo_id="Riksarkivet/satrn_htr",
    #     filename="model.pth",
    #     repo_type="model",
    #     cache_dir=".cache",
    # )

    model = Satrn(
        model="/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/.cache/models--Riksarkivet--satrn_htr/snapshots/27812c88be2706696b0283c51ee68ceb7b969301/model.pth",
        config="/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/.cache/models--Riksarkivet--satrn_htr/snapshots/27812c88be2706696b0283c51ee68ceb7b969301/config.py",
    )

    results = model([img] * 2, batch_size=2)

    print(results)
