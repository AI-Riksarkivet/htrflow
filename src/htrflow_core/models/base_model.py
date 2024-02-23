from abc import ABC, abstractmethod
from typing import Iterable, Optional
from itertools import islice

import numpy as np

from htrflow_core.results import Result


class BaseModel(ABC):

    def predict(
        self, images: Iterable[np.ndarray], batch_size: Optional[int], *args, **kwargs
    ) -> Iterable[Result]:
        """Perform inference on images

        Arguments:
            images: Input images
            batch_size: The inference batch size, optional. Will pass
                all input images at once to the model if set to None.
            *args and **kwargs: Optional arguments that are passed to
                the model specific prediction method.
        """
        out = []
        for batch in self._batch_input(images, batch_size):
            out.extend(self._predict(batch, *args, **kwargs))
        return out

    @abstractmethod
    def _predict(self, images: list[np.ndarray], *args, **kwargs) -> list[np.ndarray]:
        """Model specific prediction method"""

    def _batch_input(self, images: Iterable[np.ndarray], batch_size: Optional[int]):
        if batch_size is None:
            yield list(images)

        # TODO: Replace this routine with itertools.batch in Python 3.12
        it = iter(images)
        while batch := list(islice(it, batch_size)):
            yield batch

    def __call__(
        self,
        images: Iterable[np.ndarray],
        batch_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Iterable[Result]:
        """Alias for BaseModel.predict(...)"""
        return self.predict(images, batch_size, *args, **kwargs)
