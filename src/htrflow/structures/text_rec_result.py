from torch import Tensor

class TextRecResult():
    def __init__(self, text: str, score: Tensor):
        self.text = text
        self.score = score

