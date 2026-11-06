"""
Pipeline step for kraken text recognition.

This lives outside `pipeline.steps` so that HTRflow keeps working when kraken
is not installed. `pipeline.steps` lists it in `OPTIONAL_STEPS` and imports it
the first time a pipeline asks for it.
"""

import logging
import uuid

from htrflow.document import Document, Region, Text
from htrflow.models import get_model_by_name
from htrflow.pipeline.steps import PipelineStep, StepMetadata
from htrflow.utils.geometry import Point, Polygon


logger = logging.getLogger(__name__)


class KrakenTextRecognition(PipelineStep):
    """
    Run a kraken text recognition model.

    Unlike the generic `TextRecognition` step, which feeds a model one
    cropped line image at a time, this step reconstructs a kraken
    `Segmentation` from the document tree and hands kraken the whole page.
    kraken then extracts its own line images, straightening curved lines
    against their baselines the same way it does during training. Feeding it
    pre-cropped images instead measurably degrades accuracy on curved
    writing, which is the norm in manuscript material.

    It transcribes the same nodes the generic `TextRecognition` step would:
    every active leaf of the document tree. It is normally preceded by
    `Segmentation` with `KrakenSegmentation`, whose lines carry their
    baselines. It also works after any other segmenter, with lines at any
    depth, and on a page that was never segmented, which it reads as a single
    line. Lines without a baseline get one synthesised from their bounding
    box, though without a real baseline there is nothing to straighten.

    Example YAML:
    ```yaml
    - step: KrakenTextRecognition
      settings:
        model: KrakenRecognition
        model_settings:
          model: path/to/recognition-model.safetensors
          device: cuda
    ```

    `generation_settings` go to `KrakenRecognition.recognize`: `pad_to_width`,
    plus anything kraken's `RecognitionInferenceConfig` accepts, such as
    `batch_size`. See `KrakenRecognition` for the defaults.
    """

    def __init__(self, model, **generation_kwargs):
        """
        Arguments:
            model: An initialized `KrakenRecognition`.
            **generation_kwargs: Passed to `KrakenRecognition.recognize` on
                every page: `pad_to_width`, plus any setting kraken's
                `RecognitionInferenceConfig` accepts.
        """
        self.model = model
        self.generation_kwargs = generation_kwargs
        self.metadata = StepMetadata(str(self), self.model.metadata)

    @classmethod
    def from_config(cls, config):
        name = config.pop("model", "KrakenRecognition").lower()
        model = get_model_by_name(name)
        generation_kwargs = config.pop("generation_settings", {})
        init_kwargs = config.pop("model_settings", {}) | config
        return cls(model(**init_kwargs), **generation_kwargs)

    def run(self, document: Document) -> Document:
        image = document.image
        nodes, lines = _collect_lines(document, image.size)
        if not lines:
            logger.warning("No lines to recognize in '%s'.", document.image_name)
            return document

        segmentation = _build_segmentation(document, lines)
        records = self.model.recognize(image, segmentation, **self.generation_kwargs)

        # Match records back to nodes by line id rather than by position, so
        # that a record kraken leaves out cannot shift every later
        # transcription onto the wrong line.
        by_id = dict(zip((line.id for line in lines), nodes))
        n_attached = n_empty = 0
        for record in records:
            node = by_id.get(getattr(record, "id", None))
            if node is None:
                logger.warning("Got a transcription for an unknown line id %r.", getattr(record, "id", None))
                continue
            Text(record.prediction, confidence=_confidence(record), token_scores=_token_scores(record)).attach(node)
            n_attached += 1
            if not record.prediction:
                n_empty += 1

        if n_attached != len(nodes):
            logger.warning(
                "Got transcriptions for %d of %d lines in '%s'.", n_attached, len(nodes), document.image_name
            )
        if n_empty:
            # kraken returns an empty record, not an error, for a line it
            # could not extract, so this is the only sign of a failure.
            logger.warning(
                "%d of %d lines in '%s' came back empty. kraken does that for a line it cannot extract "
                "as well as for a blank one.",
                n_empty,
                n_attached,
                document.image_name,
            )
        return document


def _collect_lines(document: Document, size: tuple[int, int]):
    """
    Describe every line to recognize in page coordinates.

    The lines are the active leaves of the tree, which are the nodes
    `ImageLoader` hands the generic `TextRecognition` step.

    Every point is moved inside the image if it lies outside. kraken cannot
    extract a line with a point outside the image, and gives an empty
    transcription for it instead of an error. HTRflow's boxes end one pixel
    past the last column and row, so without this, any line touching the
    edge of the page would silently come back empty.

    Arguments:
        document: The document to read the lines from.
        size: The `(width, height)` of the page image kraken will see.

    Returns:
        A `(nodes, lines)` pair of parallel lists, where `lines` holds
        `kraken.containers.BaselineLine` records.
    """
    from kraken.containers import BaselineLine, BBoxLine

    width, height = size

    def clamp(points):
        return [(min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1)) for x, y in points]

    nodes, lines = [], []
    for node, origin in _active_leaves(document, Point(0, 0)):
        boundary = clamp(node.polygon.move(origin))
        baseline = node.annotations.get("baseline")

        if not baseline or len(baseline) < 2:
            # The line did not come from kraken's segmenter, so it has no
            # baseline. Synthesise one from the bounding box so that kraken
            # can still recognize the line.
            bbox = Polygon(boundary).bbox
            baseline = BBoxLine(id="_", bbox=bbox.xyxy).to_baseline().baseline

        line_id = node.annotations.get("kraken_id") or f"_{uuid.uuid4()}"
        nodes.append(node)
        lines.append(BaselineLine(id=line_id, baseline=clamp(baseline), boundary=boundary))
    return nodes, lines


def _active_leaves(node: Region, origin: Point):
    """
    Yield every active leaf under `node`, with the page position of its frame.

    HTRflow stores each polygon relative to the top-left corner of its
    parent's bounding box, so the offsets of all ancestors have to be added
    up to get back to page coordinates.

    Arguments:
        node: Where to start, normally the document.
        origin: The page coordinates of the point `node`'s polygon is
            relative to, that is, the top-left corner of its parent's
            bounding box. `Point(0, 0)` for the document.

    Yields:
        `(leaf, origin)` pairs, where `origin` is the page coordinates of the
        point the leaf's polygon is relative to.
    """
    if node.regions:
        inner = node.polygon.bbox.p1 + origin
        for child in node.regions:
            yield from _active_leaves(child, inner)
    elif node.is_active:
        yield node, origin


def _build_segmentation(document: Document, lines):
    """Wrap line records in the container kraken's recognizer expects."""
    from kraken.containers import Segmentation

    return Segmentation(
        type="baselines",
        imagename=document.image_name,
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
    )


def _confidence(record) -> float:
    """Mean per-character confidence, as a single line-level score."""
    confidences = getattr(record, "confidences", None)
    if not confidences:
        # Never return None: `Text.attach` sorts transcriptions by confidence
        # and would raise when comparing None against a float.
        return 0.0
    return float(sum(confidences) / len(confidences))


def _token_scores(record) -> list[tuple[str, float]]:
    """
    Per-character confidences, in the `(token, score)` form other HTRflow
    recognizers report per token.

    Left empty when kraken's characters and confidences do not pair up one to
    one. That happens when one of the model's output labels stands for more
    than one character, since kraken reports one confidence per label.
    """
    confidences = getattr(record, "confidences", None) or []
    if len(record.prediction) != len(confidences):
        return []
    return [(char, float(score)) for char, score in zip(record.prediction, confidences)]
