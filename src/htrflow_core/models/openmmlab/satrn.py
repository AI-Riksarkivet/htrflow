import logging

from mmocr.apis import TextRecInferencer

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import load_mmlabs
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.results import Result
from htrflow_core.utils.imgproc import NumpyImage


logger = logging.getLogger(__name__)


class Satrn(BaseModel, PytorchMixin):
    def __init__(self, model: str, config: str | None = None, device: str | None = None) -> None:
        super().__init__(device)

        model_weights, model_config = load_mmlabs(model, config)

        with SuppressOutput():
            self.model = TextRecInferencer(
                model=model_config, weights=model_weights, device=self.set_device(self.device)
            )

        logger.info(
            "Loaded Satrn model '%s' from %s with config %s on device %s",
            model,
            model_weights,
            model_config,
            self.device,
        )

        self.metadata.update(
            {
                "model": str(model),
                "config": str(config),
                "framework": Framework.Openmmlab.value,
                "task": Task.Image2Text.value,
            }
        )

    def _predict(self, images: list[NumpyImage], **kwargs) -> list[Result]:
        outputs = self.model(images, batch_size=len(images), return_datasamples=False, progress_bar=False, **kwargs)
        results = []
        for prediction in outputs["predictions"]:
            texts = prediction["text"]
            scores = prediction["scores"]
            results.append(Result.text_recognition_result(self.metadata, texts, scores))
        return results
