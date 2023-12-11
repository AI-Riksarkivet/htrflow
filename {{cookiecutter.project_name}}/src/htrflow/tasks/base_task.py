from abc import ABC, abstractmethod


# pipeline specifc
class BaseTask(ABC):

    @abstractmethod
    def preprocess():
        pass

    @abstractmethod
    def run(self, datasets):
        pass

    @abstractmethod
    def postprocess():
        pass
