import logging
import math
import os
import threading
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any, Generator, Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel as PydanticBaseModel

from htrflow import progress
from htrflow.document import Document, ImageLoader
from htrflow.models import get_model_by_name
from htrflow.pipeline.batched_queue import BatchedQueue
from htrflow.postprocess import metrics
from htrflow.postprocess.reading_order import order_regions, top_down
from htrflow.postprocess.word_segmentation import simple_word_segmentation
from htrflow.serialization import get_serializer
from htrflow.utils.imgproc import binarize
from htrflow.utils.layout import estimate_printspace, is_twopage


logger = logging.getLogger(__name__)


@dataclass
class StepMetadata:
    description: str
    settings: dict[str, str]


class PipelineStepConfig(PydanticBaseModel):
    step: str
    settings: dict | None = None


class PipelineConfig(PydanticBaseModel):
    steps: list[PipelineStepConfig]


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

    def run(self, document: Document) -> Document:
        """
        Run the pipeline step.

        Arguments:
            document: Input document

        Returns:
            A new document, updated with the results of the pipeline step.
        """

    def __str__(self):
        return self.__class__.__name__


class Inference(PipelineStep):
    """
    Run model inference.

    This is a generic pipeline step for any type of model inference.
    This step always runs the model on the images of the document's
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

    def __init__(self, model, **generation_kwargs):
        self.generation_kwargs = generation_kwargs
        self.model = model
        self.metadata = StepMetadata(str(self), self.model.metadata)

        batch_size = generation_kwargs.pop("batch_size", 1)
        self._queue = BatchedQueue(batch_size)
        self._thread = threading.Thread(target=self._process, daemon=True)
        self._thread.start()

    @classmethod
    def from_config(cls, config):
        name = config.pop("model").lower()
        model = get_model_by_name(name)
        generation_kwargs = config.pop("generation_settings", {})
        init_kwargs = config.pop("model_settings", {}) | config
        model = model(**init_kwargs)
        return cls(model, **generation_kwargs)

    def _process(self):
        while 1:
            batch = self._queue.get()
            outputs = self.model([item.item for item in batch], **self.generation_kwargs)
            for output, item in zip(outputs, batch):
                item.future.set_result(output)

    def run(self, document: Document):
        images = ImageLoader(document)
        futures = {self._queue.put(image): node for node, image in images}
        for future in as_completed(futures):
            node = futures[future]
            results = future.result()
            for result in results:
                result.attach(node)
        return document


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
    Run a text recognition model.

    See [Text recognition models](models.md#text-recognition-models) for available models.

    Example YAML:
    ```yaml
    - step: TextRecognition
      settings:
        model: TrOCR
        model_settings:
          model: Riksarkivet/trocr-base-handwritten-hist-swe-2
    ```
    """

    pass


class WordSegmentation(PipelineStep):
    """
    Segment lines into words.

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

    def run(self, document: Document):
        results = simple_word_segmentation(document.leaves())
        document.update(results)
        return document


class Export(PipelineStep):
    """
    Export results.

    Exports the current state of the document in the given format.
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

    def __init__(
        self,
        dest: str,
        format: Literal["alto", "page", "txt", "json"],
        **serializer_kwargs,
    ):
        """
        Arguments:
            dest: Output directory.
            format: Output format as a string.
        """
        self.serializer = get_serializer(format, **serializer_kwargs)
        self.dest = dest

    def run(self, document: Document):
        os.makedirs(self.dest, exist_ok=True)

        doc = self.serializer.serialize(document)
        if doc is None:
            logger.warning("Could not serialize document '%s' as %s", document.image_name, self.serializer)
            return document

        filename = os.path.join(self.dest, f"{document.image_name}{self.serializer.extension}")
        with open(filename, "w") as f:
            f.write(doc)
        logger.info("Wrote %s file to %s", self.serializer, filename)
        progress.register_export(document, filename)
        return document


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

    def run(self, document: Document):
        if document.is_leaf():
            return document

        image = document.image
        printspace = estimate_printspace(image)
        document.regions = order_regions(document.regions, printspace, self.is_twopage(image))

        for region in document.regions:
            region.regions = order_regions(region.regions, printspace, is_twopage=False)
        return document


class OrderLines(PipelineStep):
    """
    Order lines top-down.

    This step orders the lines within each region top-down.

    Example YAML:
    ```yaml
    - step: OrderLines
    ```
    """

    def run(self, document: Document):
        for node in document.traverse():
            if node.regions:
                order = top_down([region.polygon.bbox for region in node.regions])
                node.regions = [node.regions[i] for i in order]
        return document


class ExportImages(PipelineStep):
    """
    Export the document's images.

    This step writes all existing images (regions, lines, etc.) in the
    document to disk. The exported images are the images that have
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

    def run(self, document: Document):
        directory = os.path.join(self.dest, document.image_name)
        os.makedirs(directory, exist_ok=True)
        num = 1
        for image in document.segments():
            image.save(os.path.join(directory, f"image_{num:<02}.jpg"))
            num += 1
        return document


class Break(PipelineStep):
    """
    Break the pipeline! Used for testing.

    Example YAML:
    ```yaml
    - step: Break
    ```
    """

    def run(self, document):
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

    def __init__(self, condition):
        """
        Arguments:
            condition: A function `f` such that `f(node) == True` if
                `node` should be removed from the document tree.
        """
        self.condition = condition

    def run(self, document: Document):
        for node in document.traverse():
            keep = []
            for i, region in enumerate(node.regions):
                if not self.condition(region):
                    keep.append(i)
            node.regions = [node.regions[i] for i in keep]
        return document


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
        super().__init__(lambda region: metrics.text_confidence(region) < threshold)


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
        super().__init__(lambda region: metrics.average_text_confidence(region) < threshold)


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


class FilterRegionsBySize(Prune):
    """
    Filter regions by size.

    Removes all leaf nodes that are smaller or larger than the given size.

    Example YAML:
    ```yaml
    - step: FilterRegionsBySize
      settings:
        min_height: 10
        min_width: 10
        max_height: 100
        max_width: 100
    ```
    """

    def __init__(
        self, min_height: int = 0, min_width: int = 0, max_height: int | None = None, max_width: int | None = None
    ):
        """
        Arguments:
            min_height: Minimum region height in pixels.
            min_width: Minimum region width in pixels.
            max_height: Maximum region height in pixels.
            max_width: Maximum region width in pixels.
        """
        max_height = max_height or math.inf
        max_width = max_width or math.inf

        super().__init__(
            lambda node: (
                node.is_leaf()
                and not (
                    (min_height < node.polygon.height < max_height) and (min_width < node.polygon.width < max_width)
                )
            )
        )


class FilterRegionsByShape(Prune):
    """
    Filter regions by shape.

    Removes all leaf nodes that are wider or taller than the given aspect ratio(s).
    For example, if we want to filter out all regions that are more than twice as
    tall as they are wide, we set the `min_ratio` to 0.5 (1:2 width-to-height ratio).

    Example YAML:
    ```yaml
    - step: FilterRegionsByShape
      settings:
        min_ratio: 1
        max_ratio: 10
    ```
    """

    def __init__(self, min_ratio: float = 0.0, max_ratio: float = math.inf):
        """
        Arguments:
            min_ratio: Minimum width-to-height ratio.
            max_ratio: Maximum width-to-height ratio.
        """
        super().__init__(
            lambda node: node.is_leaf() and not (min_ratio < node.polygon.width / node.polygon.height < max_ratio)
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

    def run(self, document: Document):
        new_image = self.op(document.image)
        dest = os.path.join(self.output_directory)
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, document.image_name + ".jpg")
        new_image.save(path)
        return Document(path)

    def op(self, image: Image) -> Image:
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

    Runs image binarization on the document's images. Saves the
    resulting images in a directory named `binarized`. All subsequent
    pipeline steps will use the binarized images.

    Example YAML:
    ```yaml
    - step: Binarization
    ```
    """

    output_directory = "binarized"

    def op(self, image):
        return binarize(image)


def auto_import(source: list[str] | str) -> Generator[Document, Any, Any]:
    """Import document(s) from `source`

    Arguments:
        source: Import source as a single path or list of paths, where
            each path points to any of the following:
                - a directory of images
                - an image

    Yields:
        Document instances created from the given source.
    """
    paths = []
    for path in source:
        if os.path.isdir(path):
            files = [os.path.join(path, file) for file in sorted(os.listdir(path))]
            paths.extend(files)
            logger.info("Found %d files in input directory '%s'", len(files), path)
            continue
        paths.append(path)

    for path in paths:
        try:
            yield Document(path)
        except UnidentifiedImageError as e:
            logger.warning(e)
            continue


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


# Mapping class name -> class
# Ex. {segmentation: `steps.Segmentation`}
STEPS: dict[str, PipelineStep] = {cls_.__name__.lower(): cls_ for cls_ in all_subclasses(PipelineStep)}


def init_step(step: PipelineStepConfig) -> PipelineStep:
    """Initialize a pipeline step

    Arguments:
        step_name: The name of the pipeline step class. Not case sensitive.
        step_settings: A dictionary containing parameters for the step's
            __init__() method.
    """
    return STEPS[step.step.lower()].from_config(step.settings or {})
