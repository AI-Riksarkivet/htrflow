import logging
from abc import ABC, abstractmethod
from itertools import islice
from os import PathLike
from typing import Iterable, TypeVar, Union

import numpy as np
from tqdm import tqdm

from htrflow_core.models.mixins.meta_mixin import MetadataMixin
from htrflow_core.results import Result
from htrflow_core.utils import imgproc


logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class BaseModel(ABC, MetadataMixin):
    def __init__(self, device=None) -> None:
        self.device = device
        self.metadata = self.default_metadata()

    def predict(
        self, images: Iterable[np.ndarray], batch_size: int, image_scaling_factor: int = 1, *args, **kwargs
    ) -> Iterable[Result]:
        """Perform inference on images with a progress bar.

        Arguments:
            images: Input images
            batch_size: The inference batch size. Default = 1, Will pass all input images one by one to the model.
            *args and **kwargs: Optional arguments that are passed to the model specific prediction method.
        """

        tqdm_kwargs = kwargs.pop("tqdm_kwargs", {})
        tqdm_kwargs.setdefault("disable", False)
        batch_size = max(batch_size, 1)

        n_batches = (len(images) + batch_size - 1) // batch_size
        model_name = self.__class__.__name__
        logger.info(
            "Model '%s' on device '%s' received %d images in batches of %d images per batch (%d batches)",
            model_name,
            getattr(self, "device", "<device name not available>"),
            len(images),
            batch_size,
            n_batches,
        )

        results = []
        batches = self._batch_input(images, batch_size)
        desc = f"{model_name}: Running inference (batch size {batch_size})"
        progress_bar = tqdm(batches, desc, n_batches, **tqdm_kwargs)
        for i, batch in enumerate(progress_bar):
            logger.info(
                "%s: Running inference on %d images (batch %d of %d)", model_name, len(batch), i + 1, n_batches
            )
            scaled_image_batch = [imgproc.rescale_linear(image, image_scaling_factor) for image in batch]
            batch_results = self._predict(scaled_image_batch, **kwargs)
            for result in batch_results:
                if result:
                    result.rescale(1 / image_scaling_factor)
                results.append(result)
        return results

    @abstractmethod
    def _predict(self, images: list[np.ndarray], *args, **kwargs) -> list[np.ndarray]:
        """Model specific prediction method"""

    def _batch_input(self, images: Iterable[np.ndarray], batch_size: int):
        # TODO: Replace this routine with itertools.batch in Python 3.12
        it = iter(images)
        while batch := list(islice(it, batch_size)):
            yield batch

    def validate_input_before_call(self, images, images_are_nparray):
        if not isinstance(images, Iterable) or isinstance(images, (str, PathLike)):
            images = [images]
        if images_are_nparray:
            img_array = images
        else:
            img_array = [imgproc.read(img) if not isinstance(img, np.ndarray) else img for img in images]
        return img_array

    def __call__(
        self,
        images: Iterable[Union[np.ndarray, str, PathLike]],
        batch_size: int = 1,
        images_are_nparray: bool = False,
        *args,
        **kwargs,
    ) -> Iterable[Result]:
        """Alias for BaseModel.predict(...). Processes a batch of images and predicts results.

        Args:
            images (Iterable[np.ndarray]): An iterable of numpy arrays, each representing an image.
            batch_size (int): Number of images to process in a single batch.
            images_are_nparray (bool): No need to check for types in the list.
            *args, **kwargs: Additional arguments for the predict method.

        Returns:
            Iterable[Result]: The predicted results for the given images.
        """
        img_array = self.validate_input_before_call(images, images_are_nparray)

        return self.predict(img_array, batch_size, *args, **kwargs)
