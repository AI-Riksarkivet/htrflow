import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Generator, Literal

from pagexml.parser import parse_pagexml_file

from htrflow.models.base_model import BaseModel
from htrflow.models.importer import all_models
from htrflow.postprocess import metrics
from htrflow.postprocess.reading_order import order_regions, top_down
from htrflow.postprocess.word_segmentation import simple_word_segmentation
from htrflow.results import Result
from htrflow.serialization import get_serializer, save_collection
from htrflow.utils.imgproc import NumpyImage, binarize, write
from htrflow.utils.layout import estimate_printspace, is_twopage
from htrflow.volume.node import Node
from htrflow.volume.volume import Collection


logger = logging.getLogger(__name__)


@dataclass
class StepMetadata:
    description: str
    settings: dict[str, str]


class PipelineStep:
    """
    Pipeline step base class.

    Pipeline steps are implemented by subclassing this class and
    overriding the `run()` method.
    """

    parent_pipeline = None
    metadata: StepMetadata | None = None

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def run(self, collection: Collection) -> Collection:
        """
        Run the pipeline step

        Arguments:
            collection: Input collection

        Returns:
            A new collection, updated with the results of the pipeline step.
        """

    def __str__(self):
        return f"{self.__class__.__name__}"


class Inference(PipelineStep):
    """
    Run model inference

    This is a generic pipeline step for any type of model inference.
    This step always runs the model on the images of the collection's
    leaf nodes.

    Example YAML:
    ```yaml
    - step: Inference
      settings:
        model: DiT
        model_settings:
          model: ...
    ```
    """

    def __init__(self, model_class, model_kwargs, generation_kwargs):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs
        self.model = None

    def _init_model(self):
        self.model = self.model_class(**self.model_kwargs)
        self.metadata = StepMetadata(str(self), self.model.metadata)

    @classmethod
    def from_config(cls, config):
        name = config.pop("model").lower()
        if name not in MODELS:
            model_names = [model.__name__ for model in all_models()]
            msg = f"Model {name} is not supported. The available models are: {', '.join(model_names)}."
            logger.error(msg)
            raise NotImplementedError(msg)
        model = MODELS[name]
        generation_kwargs = config.pop("generation_settings", {})
        init_kwargs = config.pop("model_settings", {}) | config
        return cls(model, init_kwargs, generation_kwargs)

    def run(self, collection):
        if self.model is None:
            self._init_model()
        result = self.model(collection.segments(), **self.generation_kwargs)
        collection.update(result)
        return collection


class ImportSegmentation(PipelineStep):
    """
    Import segmentation from PageXML files.

    This step replicates the line segmentation from PageXML files.
    It can be used to import ground truth segmentation for
    evaluation purposes.

    Example YAML:
    ```yaml
    - step: ImportSegmentation
      settings:
        source: /path/to/pageXMLs
    ```
    """

    def __init__(self, source: str):
        """
        Arguments:
            source: Path to a directory with PageXML files. The XML files
                must have the same names as the input image files (ignoring
                the file extension).
        """
        self.source = source

    def run(self, collection):
        pages = []
        for page in collection:
            try:
                pages.append(parse_pagexml_file(os.path.join(self.source, page.label + ".xml")))
            except ValueError:
                pages.append(None)

        results = []
        for page in pages:
            if page is None:
                results.append(Result())
                continue
            shape = (page.coords.height, page.coords.width)
            polygons = [line.coords.points for line in page.get_lines()]
            results.append(Result.segmentation_result(shape, {}, polygons=polygons))
        collection.update(results)
        return collection


class Segmentation(Inference):
    """
    Run a segmentation model.

    See [Segmentation models](models.md#segmentation-models) for available models.

    Example YAML:
    ```yaml
    - step: Segmentation
      settings:
        model: yolo
        model_settings:
          model: Riksarkivet/yolov9-regions-1
    ```
    """

    pass


class TextRecognition(Inference):
    """
    Run a text recognition model

    See [Text recognition models](models.md#text-recognition-models) for available models.

    Example YAML:
    ```yaml
    - step: TextRecognition
      settings:
        model: TrOCR
    ```
    """

    pass


class WordSegmentation(PipelineStep):
    """
    Segment lines into words

    This step segments lines of text into words. It estimates the word
    boundaries from the recognized text, which means that this step
    must be run after a line-based text recognition model.

    See also `<models.huggingface.trocr.WordLevelTrOCR>`, which is a
    version of TrOCR that outputs word-level text directly using a more
    sophisticated method.

    Example YAML:
    ```yaml
    - step: WordSegmentation
    ```
    """

    def run(self, collection):
        results = simple_word_segmentation(collection.active_leaves())
        collection.update(results)
        return collection


class Export(PipelineStep):
    """
    Export results.

    Exports the current state of the collection in the given format.
    This step is typically the last step of a pipeline, however, it can
    be inserted at any pipeline stage. For example, you could put an
    `Export` step before a post processing step in order to save a copy
    without post processing. A pipeline can include as many `Export`
    steps as you like.

    See [Export formats](export-formats.md) or the `<serialization.serialization>`
    module for more details about each export format.

    Example:
    ```yaml
    - step: Export
      settings:
        format: Alto
        dest: alto-outputs
    ```
    """

    def __init__(self, dest: str, format: Literal["alto", "page", "txt", "json"], **serializer_kwargs):
        """
        Arguments:
            dest: Output directory.
            format: Output format as a string.
        """
        self.serializer = get_serializer(format, **serializer_kwargs)
        self.dest = dest

    def run(self, collection):
        metadata = self.parent_pipeline.metadata() if self.parent_pipeline else None
        save_collection(collection, self.serializer, self.dest, processing_steps=metadata)
        return collection


class ReadingOrderMarginalia(PipelineStep):
    """
    Order regions and lines by reading order.

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


class OrderLines(PipelineStep):
    """
    Order lines top-down.
    """

    def run(self, collection):
        for page in collection:
            for node in page.traverse():
                if node.is_region():
                    order = top_down([child.bbox for child in node])
                    node.children = [node.children[i] for i in order]
        return collection


class ExportImages(PipelineStep):
    """
    Export the collection's images.

    This step writes all existing images (regions, lines, etc.) in the
    collection to disk. The exported images are the images that have
    been passed to previous `Inference` steps and the images that would
    be passed to a following `Inference` step.

    Example YAML:
    ```yaml
    - step: ExportImages
      settings:
        dest: exported_images
    ```
    """

    def __init__(self, dest: str):
        """
        Arguments:
            dest: Destination directory.
        """
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
                write(os.path.join(directory, f"{node.label}.{extension}"), node.image)
        return collection


class Break(PipelineStep):
    """
    Break the pipeline! Used for testing.

    Example YAML:
    ```yaml
    - step: Break
    ```
    """

    def run(self, collection):
        raise Exception


class Prune(PipelineStep):
    """
    Remove nodes based on a given condition.

    This is a generic pruning (filtering) step which removes nodes
    (segments, lines, words) based on the given condition. The
    condition is a function `f` such that `f(node) == True` if `node`
    should be removed from the tree. This step runs `f` on all nodes,
    at all segmentation levels. See the `RemoveLowTextConfidence[Lines|Regions|Pages]`
    steps for examples of how to formulate `condition`.
    """

    def __init__(self, condition: Callable[[Node], bool]):
        """
        Arguments:
            condition: A function `f` such that `f(node) == True` if
                `node` should be removed from the document tree.
        """
        self.condition = condition

    def run(self, collection):
        for page in collection:
            page.prune(self.condition)
        collection.relabel()
        return collection


class RemoveLowTextConfidenceLines(Prune):
    """
    Remove all lines with text confidence score below `threshold`.

    Example YAML:
    ```yaml
    - step: RemoveLowTextConfidenceLines
      settings:
        threshold: 0.8
    ```
    """

    def __init__(self, threshold: float):
        """
        Arguments:
            threshold: Confidence score threshold.
        """
        super().__init__(lambda node: node.is_line() and metrics.line_text_confidence(node) < threshold)


class RemoveLowTextConfidenceRegions(Prune):
    """
    Remove all regions where the average text confidence score is below `threshold`.

    Example YAML:
    ```yaml
    - step: RemoveLowTextConfidenceRegions
      settings:
        threshold: 0.8
    ```
    """

    def __init__(self, threshold: float):
        """
        Arguments:
            threshold: Confidence score threshold.
        """
        super().__init__(
            lambda node: all(child.is_line() for child in node) and metrics.average_text_confidence(node) < threshold
        )


class RemoveLowTextConfidencePages(Prune):
    """
    Remove all pages where the average text confidence score is below `threshold`.

    Example YAML:
    ```yaml
    - step: RemoveLowTextConfidencePages
      settings:
        threshold: 0.8
    ```
    """

    def __init__(self, threshold: float):
        """
        Arguments:
            threshold: Confidence score threshold.
        """
        super().__init__(
            lambda node: node.parent and node.parent.is_root() and metrics.average_text_confidence(node) < threshold
        )


class ProcessImages(PipelineStep):
    """
    Base for image preprocessing steps.

    This is a base class for all image preprocessing steps. Subclasses
    define their image processing operation by overriding the `op()`
    method. This step does not alter the original image. Instead, a new
    copy of the image is saved in the directory specified by
    `ProcessImages.output_directory`. The `PageNode`'s image path is
    then updated to point to the new processed image.

    Attributes:
        output_directory: Where to write the processed images.
    """

    output_directory: str

    def run(self, collection):
        for page in collection:
            new_image = self.op(page.image)
            _, image_name = os.path.split(page.path)
            dest = os.path.join("processed_images", collection.label, self.output_directory)
            os.makedirs(dest, exist_ok=True)
            page.path = write(os.path.join(dest, image_name), new_image)
        return collection

    def op(self, image: NumpyImage) -> NumpyImage:
        """
        Perform the image processing operation on `image`.

        Arguments:
            image: Input image.

        Returns:
            A processed version of `image`.
        """
        pass


class Binarization(ProcessImages):
    """
    Binarize images.

    Runs image binarization on the collection's images. Saves the
    resulting images in a directory named `binarized`. All subsequent
    pipeline steps will use the binarized

    Example YAML:
    ```yaml
    - step: Binarization
    ```
    """

    output_directory = "binarized"

    def op(self, image):
        return binarize(image)


def auto_import(source: list[str] | str, max_size: int | None = None) -> Generator[Collection, Any, Any]:
    """Import collection(s) from `source`

    Arguments:
        source: Import source as a single path or list of paths, where
            each path points to any of the following:
                - a directory of images
                - an image
                - a pickled `Collection` instance
        max_size: The maximum number of pages in each new collection.
            Does not apply to pickled Collections.

    Yields:
        Collection instances created from the given source.
    """

    # If source is a single string, treat it as a single-item list
    # and continue
    if isinstance(source, str):
        source = [source]

    paths = []
    for path in source:
        if path.endswith("pickle"):
            yield Collection.from_pickle(path)
            continue

        if os.path.isdir(path):
            files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
            yield from _create_collection_batches(files, max_size)
            continue

        paths.append(path)
    yield from _create_collection_batches(paths, max_size)


def _create_collection_batches(paths: list[str], max_size: int | None) -> Generator[Collection, Any, Any]:
    """Create and yield collection of at most `max_size` pages"""
    if paths:
        max_size = max_size or len(paths)
        for i in range(0, len(paths), max_size):
            yield Collection(paths[i : i + max_size])


def join_collections(collections: list[Collection]) -> Collection:
    """Create a single `Collection` from the given collections."""
    label = os.path.commonprefix([col.label for col in collections])
    base = collections[0]
    for collection in collections[1:]:
        base.pages.append(collection.pages)
    base.label = label
    return base


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


# Mapping class name -> class
# Ex. {segmentation: `steps.Segmentation`}
STEPS: dict[str, PipelineStep] = {cls_.__name__.lower(): cls_ for cls_ in all_subclasses(PipelineStep)}
MODELS: dict[str, BaseModel] = {model.__name__.lower(): model for model in all_models()}


def init_step(step_name: str, step_settings: dict[str, Any]) -> PipelineStep:
    """Initialize a pipeline step

    Arguments:
        step_name: The name of the pipeline step class. Not case sensitive.
        step_settings: A dictionary containing parameters for the step's
            __init__() method.
    """
    return STEPS[step_name.lower()].from_config(step_settings)
