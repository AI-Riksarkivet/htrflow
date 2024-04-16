from enum import Enum


class Framework(Enum):
    """Frameworks describes which framework is used in the implementation of the model"""

    Openmmlab: str = "openmmlab"
    HuggingFace: str = "huggingface"
    Ultralytics: str = "ultralytics"


class Task(Enum):
    """Task describe the “shape” of each model inputs and outputs"""

    Image2Text: str = "image_2_text"
    ObjectDetection: str = "object_detection"
    InstanceSegmentation: str = "instance_segmentation"
    ImageClassification: str = "image_classification"
