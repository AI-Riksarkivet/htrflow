from abc import ABC, abstractmethod
from itertools import islice
from os import PathLike
from typing import Iterable, Union

import numpy as np
from tqdm import tqdm

from htrflow_core.models.meta_mixin import MetadataMixin
from htrflow_core.results import Result
from htrflow_core.utils import imgproc


class BaseModel(ABC, MetadataMixin):
    def __init__(self, **kwargs) -> None:
        self.device = kwargs.get("device", None)
        self.cache_dir = kwargs.get("cache_dir", "./.cache")
        self.metadata = self.default_metadata()

    def predict(self, images: Iterable[np.ndarray], batch_size: int, *args, **kwargs) -> Iterable[Result]:
        """Perform inference on images with a progress bar.

        Arguments:
            images: Input images
            batch_size: The inference batch size. Default = 1, Will pass all input images one by one to the model.
            *args and **kwargs: Optional arguments that are passed to the model specific prediction method.
        """
        out = []

        tqdm_kwargs = kwargs.pop("tqdm_kwargs", {})
        tqdm_kwargs.setdefault("disable", False)

        for batch in tqdm(
            self._batch_input(images, batch_size),
            total=self._tqdm_total(images, batch_size),
            desc=self._tqdm_description(batch_size),
            **tqdm_kwargs,
        ):
            out.extend(self._predict(batch, *args, **kwargs))
        return out

    @abstractmethod
    def _predict(self, images: list[np.ndarray], *args, **kwargs) -> list[np.ndarray]:
        """Model specific prediction method"""

    def _tqdm_description(self, batch_size: int) -> str:
        model_name = self.__class__.__name__
        tqdm_description = (
            f"{model_name}: Running batch inference" if batch_size > 1 else f"{model_name}: Running inference"
        )
        return tqdm_description

    def _tqdm_total(self, images, batch_size: int) -> int:
        return (len(images) + batch_size - 1) // batch_size

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
