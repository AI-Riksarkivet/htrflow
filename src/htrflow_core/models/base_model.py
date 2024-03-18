from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

from htrflow_core.results import Result


class BaseModel(ABC):
    def __init__(self, device: Optional[str] = None):
        self.model = None
        self._initialize_device(device)

    def _initialize_device(self, device: Optional[str]):
        """Private method to initialize the device."""
        self._device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: Optional[str]):
        self._initialize_device(value)

    def to(self, device: Optional[str]):
        self.device = device

    def predict(self, images: Iterable[np.ndarray], batch_size: int, *args, **kwargs) -> Iterable[Result]:
        """Perform inference on images with a progress bar.

        Arguments:
            images: Input images
            batch_size: The inference batch size. Default = 1, Will pass all input images one by one to the model.
            *args and **kwargs: Optional arguments that are passed to
                the model specific prediction method.
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

    def to_numpy(self, tensor):
        """
        Convert a PyTorch tensor to a NumPy array.
        Moves the tensor to CPU if it's on a GPU.
        """
        # Check if the tensor is on a CUDA device and move it to CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()

    def __call__(
        self,
        images: Iterable[np.ndarray],
        batch_size: int = 1,
        *args,
        **kwargs,
    ) -> Iterable[Result]:
        """Alias for BaseModel.predict(...)"""
        return self.predict(images, batch_size, *args, **kwargs)
