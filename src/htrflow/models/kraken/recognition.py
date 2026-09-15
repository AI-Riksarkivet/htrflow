"""
HTRflow adapter of kraken's text recognizers.
"""

import logging
from typing import Any

from PIL import Image

from htrflow.models.base_model import BaseModel
from htrflow.models.kraken.resolve import resolve_kraken_model


logger = logging.getLogger(__name__)


class KrakenRecognition(BaseModel):
    """
    HTRflow adapter of kraken's text recognition models.

    This model is driven by the `KrakenTextRecognition` pipeline step rather
    than the generic `Inference` step, and it deliberately does not implement
    the usual "list of line images in, list of transcriptions out" contract.

    kraken recognizers are trained on line images that kraken extracts itself
    from a page plus a baseline, straightening curved lines in the process.
    Handing them the polygon-masked bounding-box crop that `Inference` would
    produce skips that step and costs accuracy on curved writing. The step
    therefore passes the whole page and lets kraken do its own extraction.

    Example usage with the `KrakenTextRecognition` step:
    ```yaml
    - step: KrakenTextRecognition
      settings:
        model: KrakenRecognition
        model_settings:
          model: validation/model/ppocrv6-medium.safetensors
          device: cuda
        generation_settings:
          batch_size: 8
    ```

    `model` accepts a local path, an htrmopo DOI such as
    `10.5281/zenodo.21788409`, or a Hugging Face repo ID.
    """

    def __init__(self, model: str, revision: str | None = None, **kwargs):
        """
        Arguments:
            model: Path to a kraken recognition model, an htrmopo DOI, or a
                Hugging Face repo ID.
            revision: Optional revision of the Hugging Face repository.
            **kwargs: Forwarded to `BaseModel.__init__`, notably `device`.
        """
        super().__init__(**kwargs)

        from kraken.tasks import RecognitionTaskModel

        model_path, model_version = resolve_kraken_model(model, revision)
        self.model = RecognitionTaskModel.load_model(model_path)

        logger.info(
            "Initialized kraken recognition model '%s' (%s, seg_type=%s) from %s on device %s",
            model, type(self.model.net).__name__, self.model.seg_type, model_path, self.device,
        )
        self.metadata.update(
            {
                "model": model,
                "model_version": model_version,
                "architecture": type(self.model.net).__name__,
            }
        )

    def _predict(self, images: list[Image.Image], **kwargs):
        """Not supported -- see `recognize`."""
        raise NotImplementedError(
            "KrakenRecognition does not support the generic `Inference` step, because "
            "kraken extracts its own line images from the page. Use the "
            "`KrakenTextRecognition` step instead of `TextRecognition`."
        )

    def recognize(self, image: Image.Image, segmentation, **config_kwargs):
        """
        Transcribe every line of `segmentation` on `image`.

        Arguments:
            image: The full page image.
            segmentation: A `kraken.containers.Segmentation` covering it.
            **config_kwargs: Forwarded to kraken's
                `RecognitionInferenceConfig`, for example `batch_size`,
                `temperature` or `padding`.

        Returns:
            One `ocr_record` per line, in the order of `segmentation.lines`.
        """
        from kraken.configs import RecognitionInferenceConfig

        config = RecognitionInferenceConfig(**self._inference_settings(config_kwargs))
        return list(self.model.predict(image, segmentation, config))

    def _inference_settings(self, config_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Translate HTRflow's torch device into kraken's Lightning settings.

        `num_line_workers` defaults to 0 rather than kraken's 2: the worker
        pool forks, which deadlocks on some CUDA setups, and line extraction
        is not the bottleneck for page-at-a-time inference. Raise it in the
        YAML if your machine is happy with it.
        """
        settings = {"num_line_workers": 0} | dict(config_kwargs)
        if "accelerator" not in settings:
            settings["accelerator"] = "gpu" if self.device.type == "cuda" else "cpu"
        if "device" not in settings:
            settings["device"] = [self.device.index] if self.device.index is not None else 1
        return settings
