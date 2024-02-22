from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from htrflow_core.results import Result


# Implemtnaton specfic for inferencers
class BaseInferencer(ABC):
    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def predict(self, input_image):
        pass

    @abstractmethod
    def postprocess(self):
        pass


class BaseModel(ABC):

    @abstractmethod
    def predict(self, images: Iterable[np.ndarray], *args, **kwargs) -> Iterable[Result]:
        pass

    def __call__(self, images: Iterable[np.ndarray], *args, **kwargs) -> Iterable[Result]:
        return self.predict(images, *args, **kwargs)
