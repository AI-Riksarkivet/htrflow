from abc import ABC, abstractmethod


# pipeline specifc
class BaseTask(ABC):

    @abstractmethod
    def run(self):
        pass
