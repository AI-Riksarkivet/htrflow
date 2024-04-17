import logging
import os

from htrflow_core.dummies.dummy_models import simple_word_segmentation
from htrflow_core.models.importer import all_models
from htrflow_core.serialization import get_serializer
from htrflow_core.utils.imgproc import binarize
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
        if name not in MODELS:
            model_names = [model.__name__ for model in all_models()]
            msg = f"Model {name} is not supported. The available models are: {', '.join(model_names)}."
            raise NotImplementedError(msg)
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


class Binarization(PipelineStep):
    def run(self, volume):
        for page in volume:
            page.image = binarize(page.image)
        return volume


class WordSegmentation(PipelineStep):

    requires = [TextRecognition]

    def run(self, volume):
        results = simple_word_segmentation(volume.leaves())
        volume.update(results)
        return volume


class Export(PipelineStep):
    def __init__(self, dest, format, **serializer_kwargs):
        self.serializer = get_serializer(format, **serializer_kwargs)
        self.dest = dest

    def run(self, volume):
        volume.save(self.dest, self.serializer)
        return volume


def auto_import(source) -> Volume:
    """Import volume from `source`

    Automatically detects import type from the input. Supported types
    are:
        - A path to a directory with images
        - A list of paths to images
        - A volume instance (returns itself)
    """
    if isinstance(source, Volume):
        return source
    elif isinstance(source, list):
        return Volume(source)
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
MODELS = {model.__name__.lower(): model for model in all_models()}


def init_step(step):
    name = step["step"].lower()
    config = step.get("settings", {})
    return STEPS[name].from_config(config)
