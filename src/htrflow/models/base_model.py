from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self):
        self._model
        self.framework
        self.device

    @abstractmethod
    def predict(self):
        pass

    @property
    @abstractmethod
    def model(self):
        return self._model
