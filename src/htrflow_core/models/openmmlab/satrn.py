import logging

from mmocr.apis import TextRecInferencer

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.hf_utils import load_mmlabs, commit_hash_from_path
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.results import Result
from htrflow_core.utils.imgproc import NumpyImage


logger = logging.getLogger(__name__)


class Satrn(BaseModel):
    """
    HTRFLOW adapter of Openmmlabs' Satrn model
    """

    def __init__(self, model: str, config: str | None = None, device: str | None = None) -> None:
        """
        Initialize a Satrn model.

        Arguments:
            model: Path to a local .pth model weights file or to a
                huggingface repo which contains a .pth file, for example
                'Riksarkivet/satrn_htr'.
            config: Path to a local config.py file or to a huggingface
                repo which contains a config.py file, for example
                'Riksarkivet/satrn_htr'.
            device: Model device.
        """
        super().__init__(device)

        config = config or model
        model_weights, model_config = load_mmlabs(model, config)

        with SuppressOutput():
            self.model = TextRecInferencer(
                model=model_config, weights=model_weights, device=self.device
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
                "model": model,
                "model_version": commit_hash_from_path(model_weights),
                "config": config,
                "config_version": commit_hash_from_path(model_config),
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
