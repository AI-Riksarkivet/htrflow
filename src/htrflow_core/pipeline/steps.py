import logging
import os

from htrflow_core.dummies.dummy_models import RecognitionModel, SegmentationModel, simple_word_segmentation
from htrflow_core.volume.volume import Volume


logger = logging.getLogger(__name__)


class PipelineStep:
    """Pipeline step base class

    Class attributes:
        requires: A list of steps that need to precede this step.
    """

    requires = []

    def run(self, volume: Volume) -> Volume:
        """Run step"""

    def __str__(self):
        return f"{self.__class__.__name__}"


class Inference(PipelineStep):

    def __init__(self, model):
        self.model = model

    def run(self, volume):
        result = self.model(volume.segments())
        volume.update(result)
        return volume


class Segmentation(Inference):
    def __init__(self):
        super().__init__(SegmentationModel())


class TextRecognition(Inference):
    def __init__(self):
        super().__init__(RecognitionModel())


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


def init_step(step):
    name = step["step"].lower()
    kwargs = step.get("settings", {})
    return STEPS[name](**kwargs)
