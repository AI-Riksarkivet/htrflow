import logging
import os

# Imported-but-unused models are needed here in order for
# `all_subclasses` to find them
from htrflow_core.dummies.dummy_models import (  # noqa: F401
    RecognitionModel,
    SegmentationModel,
    simple_word_segmentation,
)
from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.huggingface.trocr import TrOCR  # noqa: F401
from htrflow_core.models.ultralytics.yolo import YOLO  # noqa: F401
from htrflow_core.volume.volume import Volume


logger = logging.getLogger(__name__)


class PipelineStep:
    """Pipeline step base class

    Class attributes:
        requires: A list of steps that need to precede this step.
    """

    requires = []

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def run(self, volume: Volume) -> Volume:
        """Run step"""

    def __str__(self):
        return f"{self.__class__.__name__}"


class Inference(PipelineStep):

    def __init__(self, model, generation_kwargs):
        self.model = model
        self.generation_kwargs = generation_kwargs

    @classmethod
    def from_config(cls, config):
        name = config["model"].lower()
        init_kwargs = config.get("model_settings", {})
        model = MODELS[name](**init_kwargs)
        generation_kwargs = config.get("generation_settings", {})
        return cls(model, generation_kwargs)

    def run(self, volume):
        result = self.model(volume.segments(), **self.generation_kwargs)
        volume.update(result)
        return volume


class Segmentation(Inference):
    pass


class TextRecognition(Inference):
    pass


class WordSegmentation(PipelineStep):

    requires = [TextRecognition]

    def run(self, volume):
        results = simple_word_segmentation(volume.leaves())
        volume.update(results)
        return volume


def auto_import(source) -> Volume:
    """Import volume from `source`

    Automatically detects import type from the input. Supported types
    are:
        - directories with images
    """
    if isinstance(source, Volume):
        return source
    elif isinstance(source, str):
        if os.path.isdir(source):
            logger.info("Loading volume from directory %s", source)
            return Volume.from_directory(source)
    raise ValueError(f"Could not infer import type for '{source}'")


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])

# Mapping class name -> class
# Ex. {segmentation: `steps.Segmentation`}
STEPS = {cls_.__name__.lower(): cls_ for cls_ in all_subclasses(PipelineStep)}
MODELS = {cls_.__name__.lower(): cls_ for cls_ in all_subclasses(BaseModel)}


def init_step(step):
    name = step["step"].lower()
    config = step.get("settings", {})
    return STEPS[name].from_config(config)
