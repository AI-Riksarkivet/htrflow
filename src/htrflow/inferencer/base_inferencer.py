from abc import ABC, abstractmethod


class BaseInferencer(ABC):
    @abstractmethod
    def preprocess():
        pass

    @abstractmethod
    def predict(self, input_image):
        pass

    @abstractmethod
    def postprocess():
        pass
