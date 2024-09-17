import logging
from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Collection, Generator, Iterable, TypeVar

import torch
from tqdm import tqdm

from htrflow.results import Result
from htrflow.utils.imgproc import NumpyImage, rescale_linear


logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class BaseModel(ABC):
    """
    Model base class

    This is the abstract base class of HTRFlow models. It handles batching
    of inputs, some shared initialization arguments and generic logging.

    Concrete model implementations bases this class and defines their
    prediction method in `_predict()`.
    """

    def __init__(self, device: str | None = None, allow_tf32: bool = True):
        """
        Arguments:
            device: Model device as a string, recognizable by torch. Defaults
                to `None`, which sets the device to `cuda` or `cpu` depending
                on availability.
            allow_tf32: Allow running matrix multiplications with TensorFloat-32.
                This speeds up inference at the expense of inference quality.
                Read more here:
                https://huggingface.co/docs/diffusers/optimization/fp16#tensorfloat-32
        """
        self.metadata = {"model_class": self.__class__.__name__}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Allow matrix multiplication with TensorFloat-32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    def predict(
        self,
        images: Collection[NumpyImage],
        batch_size: int = 1,
        image_scaling_factor: float = 1.0,
        tqdm_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[Result]:
        """Perform inference on images

        Takes an arbitrary number of inputs and runs batched inference.
        The inputs can be streamed from an iterator and don't need to
        be simultaneously read into memory. Prints a progress bar using
        `tqdm`. This is a template method which uses the model-specific
        `_predict(...)`.

        Arguments:
            images: Input images
            batch_size: Inference batch size, defaults to 1
            image_scaling_factor: If < 1, all input images will be down-
                scaled by this factor, which can be useful for speeding
                up inference on higher resolution images. All geometric
                data in the result (e.g., bounding boxes) are reported
                with respect to the original resolution.
            tqdm_kwargs: Optional keyword arguments to control the
                progress bar.
            **kwargs: Optional keyword arguments that are forwarded to
                the model specific prediction method `_predict(...)`.
        """

        batch_size = max(batch_size, 1)  # make sure batch size is at least 1
        image_scaling_factor = max(10e-10, min(image_scaling_factor, 1))  # clip scaling factor to (0, 1]

        n_batches = (len(images) + batch_size - 1) // batch_size
        model_name = self.__class__.__name__
        logger.info(
            "Model '%s' on device '%s' received %d images in batches of %d images per batch (%d batches)",
            model_name,
            self.device,
            len(images),
            batch_size,
            n_batches,
        )

        results = []
        batches = _batch(images, batch_size)
        desc = f"{model_name}: Running inference (batch size {batch_size})"
        for i, batch in enumerate(tqdm(batches, desc, n_batches, **(tqdm_kwargs or {}))):
            msg = "%s: Running inference on %d images (batch %d of %d)"
            logger.info(msg, model_name, len(batch), i + 1, n_batches)
            scaled_batch = [rescale_linear(image, image_scaling_factor) for image in batch]
            batch_results = self._predict(scaled_batch, **kwargs)
            for result in batch_results:
                result.rescale(1 / image_scaling_factor)
                results.append(result)
        return results

    @abstractmethod
    def _predict(self, images: list[NumpyImage], **kwargs) -> list[Result]:
        """Model specific prediction method"""

    def __call__(self, images: Collection[NumpyImage], **kwargs) -> list[Result]:
        """Alias for BaseModel.predict(...)"""
        return self.predict(images, **kwargs)


def _batch(iterable: Iterable[_T], batch_size: int) -> Generator[list[_T], None, None]:
    """Yield fixed-size batches from `iterable`"""
    # TODO: Replace this routine with itertools.batch in Python 3.12
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch
