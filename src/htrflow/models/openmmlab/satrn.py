import logging

from mmocr.apis import TextRecInferencer

from htrflow.models.base_model import BaseModel
from htrflow.models.hf_utils import commit_hash_from_path, load_mmlabs
from htrflow.models.openmmlab.utils import SuppressOutput
from htrflow.results import Result
from htrflow.utils.imgproc import NumpyImage


logger = logging.getLogger(__name__)


class Satrn(BaseModel):
    """
    HTRFlow adapter of Openmmlabs' Satrn model

    Example usage with the `TextRecognition` pipeline step:
    ```yaml
    - step: TextRecognition
      settings:
        model: Satrn
        model_settings:
          model: Riksarkivet/satrn_htr
    ```
    """

    def __init__(
        self,
        model: str,
        config: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> None:
        """
        Arguments:
            model: Path to a local .pth model weights file or to a
                huggingface repo which contains a .pth file, for example
                'Riksarkivet/satrn_htr'.
            config: Path to a local config.py file or to a huggingface
                repo which contains a config.py file, for example
                'Riksarkivet/satrn_htr'.
            kwargs: Additional kwargs which are forwarded to BaseModel's
                `__init__`.
        """
        super().__init__(**kwargs)

        config = config or model
        model_weights, model_config = load_mmlabs(model, config, revision)

        with SuppressOutput():
            self.model = TextRecInferencer(model=model_config, weights=model_weights, device=self.device)

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
        """
        Satrn-specific prediction method

        This method is used by `predict()` and should typically not be
        called directly.

        Arguments:
            images: Input images
            kwargs: Additional keyword arguments that are forwarded to
                `mmocr.apis.TextRecInferencer.__call__()`.
        """
        outputs = self.model(images, batch_size=len(images), return_datasamples=False, progress_bar=False, **kwargs)
        results = []
        for prediction in outputs["predictions"]:
            texts = prediction["text"]
            scores = prediction["scores"]
            results.append(Result.text_recognition_result(self.metadata, texts, scores))
        return results
