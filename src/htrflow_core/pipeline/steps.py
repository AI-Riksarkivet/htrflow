import logging
import os
from typing import Literal

from htrflow_core.models.importer import all_models
from htrflow_core.postprocess.reading_order import order_regions
from htrflow_core.postprocess.word_segmentation import simple_word_segmentation
from htrflow_core.serialization import get_serializer
from htrflow_core.utils.imgproc import write
from htrflow_core.utils.layout import estimate_printspace, is_twopage
from htrflow_core.volume.volume import Collection


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

    def run(self, collection: Collection) -> Collection:
        """Run step"""

    def __str__(self):
        return f"{self.__class__.__name__}"


class Inference(PipelineStep):
    def __init__(self, model_class, model_kwargs, generation_kwargs):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self.model = None

    def _init_model(self):
        self.model = self.model_class(**self.model_kwargs)

    @classmethod
    def from_config(cls, config):
        name = config["model"].lower()
        if name not in MODELS:
            model_names = [model.__name__ for model in all_models()]
            msg = f"Model {name} is not supported. The available models are: {', '.join(model_names)}."
            logger.error(msg)
            raise NotImplementedError(msg)
        init_kwargs = config.get("model_settings", {})
        model = MODELS[name]
        generation_kwargs = config.get("generation_settings", {})
        return cls(model, init_kwargs, generation_kwargs)

    def run(self, collection):
        if self.model is None:
            self._init_model()
        result = self.model(collection.segments(), **self.generation_kwargs)
        collection.update(result)
        return collection


class Segmentation(Inference):
    pass


class TextRecognition(Inference):
    pass


class WordSegmentation(PipelineStep):
    requires = [TextRecognition]

    def run(self, collection):
        results = simple_word_segmentation(collection.active_leaves())
        collection.update(results)
        return collection


class Export(PipelineStep):
    def __init__(self, dest, format, **serializer_kwargs):
        self.serializer = get_serializer(format, **serializer_kwargs)
        self.dest = dest

    def run(self, collection):
        collection.save(self.dest, self.serializer)
        return collection


class ReadingOrderMarginalia(PipelineStep):
    """Apply reading order

    This step orders the pages' first- and second-level segments
    (corresponding to regions and lines). Both the regions and their
    lines are ordered using `reading_order.order_regions`.
    """
    def __init__(self, two_page: Literal["auto"] | bool = False):
        """
        Arguments:
            two_page: Whether the page is a two-page spread. Three modes:
                - 'auto': determine heuristically for each page using
                    `layout.is_twopage`
                - True: assume all pages are spreads
                - False: assume all pages are single pages
        """
        self.two_page = two_page

    def is_twopage(self, image):
        if self.two_page == "auto":
            return is_twopage(image)
        return self.two_page

    def run(self, collection):
        for page in collection:
            if page.is_leaf():
                continue

            image = page.image
            printspace = estimate_printspace(image)
            page.children = order_regions(page.children, printspace, self.is_twopage(image))

            for region in page:
                region.children = order_regions(region.children, printspace, is_twopage=False)
        collection.relabel()
        return collection


class ExportImages(PipelineStep):
    """Export the collection's images

    This step writes all existing images (regions, lines, etc.) in the
    collection to disk.
    """

    def __init__(self, dest):
        self.dest = dest
        os.makedirs(self.dest, exist_ok=True)

    def run(self, collection):
        for page in collection:
            directory = os.path.join(self.dest, page.get("image_name"))
            extension = page.get("image_path").split(".")[-1]
            os.makedirs(directory, exist_ok=True)
            for node in page.traverse():
                if node.image is None:
                    continue
                write(os.path.join(directory, f'{node.label}.{extension}'), node.image)
        return collection


class Break(PipelineStep):
    """Break the pipeline! Used for testing."""

    def run(self, collection):
        raise Exception


def auto_import(source: Collection | list[str] | str) -> Collection:
    """Import collection from `source`

    Automatically detects import type from the input. Supported types
    are:
        - A path to a directory with images
        - A list of paths to images
        - A path to a pickled collection
        - A collection instance (returns itself)
    """
    if isinstance(source, Collection):
        return source

    # If source is a single string, treat it as a single-item list
    # and continue
    if isinstance(source, str):
        source = [source]

    if isinstance(source, list):
        # Input is a single directory
        if len(source) == 1:
            if os.path.isdir(source[0]):
                logger.info("Loading collection from directory %s", source[0])
                return Collection.from_directory(source[0])
            if source[0].endswith("pickle"):
                return Collection.from_pickle(source[0])

        # Input is a list of (potential) file paths, check each and
        # keep only the ones that refers to files
        paths = []
        for path in source:
            if not os.path.isfile(path):
                logger.info("Skipping %s, not a regular file", path)
                continue
            paths.append(path)

        if paths:
            logger.info("Loading collection from %d file(s)", len(paths))
            return Collection(paths)

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
