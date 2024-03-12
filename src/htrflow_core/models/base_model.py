from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from tqdm import tqdm

from htrflow_core.results import Result


class BaseModel(ABC):
    def _device(self, device: Optional[str]):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def to(self, device: Optional[str]):
        self._device(device)
        if self.model:
            self.model.to(device)

    def predict(self, images: Iterable[np.ndarray], batch_size: int, *args, **kwargs) -> Iterable[Result]:
        """Perform inference on images with a progress bar.

        Arguments:
            images: Input images
            batch_size: The inference batch size. Default = 1, Will pass all input images one by one to the model.
            *args and **kwargs: Optional arguments that are passed to
                the model specific prediction method.
        """

        out = []
        batch_generator = BatchGenerator(images, batch_size, self.__class__.__name__)

        for batch in tqdm(
            batch_generator,
            total=len(batch_generator),
            desc=batch_generator.tqdm_description(),
            disable=False,  # Control verbosity
        ):
            out.extend(self._predict(batch, *args, **kwargs))
        return out

    @abstractmethod
    def _predict(self, images: list[np.ndarray], *args, **kwargs) -> list[np.ndarray]:
        """Model specific prediction method"""

    def _tqdm_total(self, images, batch_size: int) -> int:
        return (len(images) + batch_size - 1) // batch_size

    # def _batch_input(self, images: Iterable[np.ndarray], batch_size: int):
    #     # TODO: Replace this routine with itertools.batch in Python 3.12
    #     it = iter(images)
    #     while batch := list(islice(it, batch_size)):
    #         yield batch

    def __call__(
        self,
        images: Iterable[np.ndarray],
        batch_size: int = 1,
        *args,
        **kwargs,
    ) -> Iterable[Result]:
        """Alias for BaseModel.predict(...)"""
        return self.predict(images, batch_size, *args, **kwargs)


class BatchGenerator:
    """Batch generator that also supports len() and a custom tqdm description for progress."""

    def __init__(self, images: Iterable[np.ndarray], batch_size: int, model_name: str):
        self.images = images
        self.batch_size = batch_size
        self.model_name = model_name
        self.total_batches = self._calculate_total_batches(images, batch_size)

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        """Generates batches of images."""
        it = iter(self.images)
        while batch := list(islice(it, self.batch_size)):
            yield batch

    def tqdm_description(self) -> str:
        """Generate a tqdm description based on the batch size."""
        return (
            f"{self.model_name}: Running batch inference"
            if self.batch_size > 1
            else f"{self.model_name}: Running inference"
        )

    @staticmethod
    def _calculate_total_batches(images, batch_size: int) -> int:
        """Calculate the total number of batches."""
        return (len(images) + batch_size - 1) // batch_size
