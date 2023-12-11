from abc import ABC, abstractmethod


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

