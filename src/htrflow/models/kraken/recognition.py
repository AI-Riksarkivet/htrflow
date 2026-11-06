"""
HTRflow adapter of kraken's text recognizers.
"""

import logging
from contextlib import contextmanager
from typing import Any

from PIL import Image

from htrflow.models.base_model import BaseModel
from htrflow.models.kraken.resolve import resolve_kraken_model, wrong_task_message


logger = logging.getLogger(__name__)

# How torch device types map onto kraken's Lightning accelerators.
_ACCELERATORS = {"cuda": "gpu", "mps": "mps"}


def lightning_settings(device, settings: dict[str, Any]) -> dict[str, Any]:
    """
    Fill in kraken's Lightning `accelerator` and `device` from a torch device.

    kraken runs inference through Lightning, which takes an accelerator name
    plus a list of device indices (or a device count) rather than a torch
    device. Values already in `settings`, for example from the YAML, are kept.
    """
    settings = dict(settings)
    settings.setdefault("accelerator", _ACCELERATORS.get(device.type, "cpu"))
    settings.setdefault("device", [device.index] if device.index is not None else 1)
    return settings


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
          model: path/to/recognition-model.safetensors
          device: cuda
    ```

    `model` accepts a local path, an htrmopo DOI, or a Hugging Face repo ID.
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
        try:
            self.model = RecognitionTaskModel.load_model(model_path)
        except ValueError as e:
            raise ValueError(wrong_task_message(model_path, model, "recognition")) from e

        logger.info(
            "Initialized kraken recognition model '%s' (%s, seg_type=%s) from %s on device %s",
            model,
            type(self.model.net).__name__,
            self.model.seg_type,
            model_path,
            self.device,
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

    def recognize(self, image: Image.Image, segmentation, pad_to_width: int | None = None, **config_kwargs):
        """
        Transcribe every line of `segmentation` on `image`.

        Arguments:
            image: The full page image.
            segmentation: A `kraken.containers.Segmentation` covering it.
            pad_to_width: Pad every batch with blank space to at least this
                width, in pixels of the scaled line images the network sees.
                kraken pads a batch only to its widest line, but a model
                trained on lines padded to a fixed width is more accurate when
                inference pads the same way. kraken's PP-OCRv6 trainer pads to
                2,560 by default. Off unless set.
            **config_kwargs: Forwarded to kraken's
                `RecognitionInferenceConfig`, for example `batch_size`,
                `temperature` or `padding`.

        Returns:
            One `ocr_record` per line, in the order of `segmentation.lines`.
        """
        from kraken.configs import RecognitionInferenceConfig

        config = RecognitionInferenceConfig(**self._inference_settings(config_kwargs))
        with _batches_padded_to(self.model.net, pad_to_width):
            return list(self.model.predict(image, segmentation, config))

    def _inference_settings(self, config_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Combine HTRflow's defaults, the YAML's `generation_settings` and the
        model's device into settings for kraken's `RecognitionInferenceConfig`.

        Two defaults differ from kraken's own.

        `batch_size` is 32 rather than 1. kraken pads every batch to its widest
        line, so a batch of one is never padded at all, and recognizers trained
        on padded lines lose accuracy on it. Measured on 199 out-of-domain
        lines with a PP-OCRv6 recognizer, a batch of one scored 6.65% CER
        against 5.97% for batches of 32. Batching is also simply faster.

        `num_line_workers` is 0 rather than kraken's 2: the worker pool forks,
        which deadlocks on some CUDA setups, and line extraction is not the
        bottleneck for page-at-a-time inference.

        Both can be overridden from `generation_settings` in the YAML.
        """
        return lightning_settings(self.device, {"num_line_workers": 0, "batch_size": 32} | dict(config_kwargs))


@contextmanager
def _batches_padded_to(net, width: int | None):
    """
    Make kraken pad every batch to at least `width` while the block runs.

    kraken pads a batch to its widest line, then hands it to the network
    through `_rec_predict(seqs, seq_lens)`. Padding further at that point
    reproduces fixed-width training: the padding is zeros like kraken's own,
    and `seq_lens` keeps every line's true width.

    `_rec_predict` ships with kraken, but its leading underscore marks it as
    internal: kraken does not promise to keep it unchanged between releases.
    This is the one place the wrapper relies on such a method. It is
    replaced on this one network object only, and restored afterwards. If a
    kraken release removes it, `pad_to_width` raises an error. A unit test
    runs this against kraken's real recognition code, so a release that
    changes how batches reach the network fails the tests instead of quietly
    disabling the padding.
    """
    if not width:
        yield
        return

    original = getattr(net, "_rec_predict", None)
    if original is None:
        raise ValueError(f"pad_to_width is not supported for {type(net).__name__} models.")

    import torch.nn.functional as F

    def padded(seqs, seq_lens):
        if seqs.shape[-1] < width:
            seqs = F.pad(seqs, (0, width - seqs.shape[-1]))
        return original(seqs, seq_lens)

    # Normally `_rec_predict` comes from the network's class, and deleting
    # our override afterwards makes the class method visible again. If the
    # object already had its own `_rec_predict`, put that back instead.
    patched_before = "_rec_predict" in vars(net)
    net._rec_predict = padded
    try:
        yield
    finally:
        if patched_before:
            net._rec_predict = original
        else:
            del net._rec_predict
