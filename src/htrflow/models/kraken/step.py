"""
Pipeline step for kraken text recognition.

This lives outside `pipeline.steps` so that HTRflow keeps working when kraken
is not installed. It is registered by an import in `pipeline.steps`, which
picks up the subclass through `PipelineStep.__subclasses__`.
"""

import logging
import uuid

from htrflow.document import Document, Text
from htrflow.models import get_model_by_name
from htrflow.pipeline.steps import PipelineStep, StepMetadata
from htrflow.utils.geometry import Polygon


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

    It is normally preceded by `Segmentation` with `KrakenSegmentation`,
    which stores the baseline of every line. It also works after any other
    segmenter: lines without a baseline get one synthesised from their
    bounding box, though without a real baseline there is nothing to
    straighten.

    Example YAML:
    ```yaml
    - step: KrakenTextRecognition
      settings:
        model: KrakenRecognition
        model_settings:
          model: validation/model/ppocrv6-medium.safetensors
          device: cuda
        generation_settings:
          batch_size: 8
    ```
    """

    def __init__(self, model, **generation_kwargs):
        """
        Arguments:
            model: An initialized `KrakenRecognition`.
            **generation_kwargs: Forwarded to kraken's
                `RecognitionInferenceConfig`.
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
        nodes, lines = _collect_lines(document)
        if not lines:
            logger.warning("No lines to recognize in '%s'.", document.image_name)
            return document

        segmentation = _build_segmentation(document, lines)
        records = self.model.recognize(document.image, segmentation, **self.generation_kwargs)

        # Match records back to nodes by line id. kraken yields records in
        # segmentation order, but a line whose extraction fails is dropped,
        # which would silently shift every following transcription by one.
        by_id = dict(zip((line.id for line in lines), nodes))
        n_attached = 0
        for record in records:
            node = by_id.get(getattr(record, "id", None))
            if node is None:
                logger.warning("Got a transcription for an unknown line id %r.", getattr(record, "id", None))
                continue
            Text(record.prediction, confidence=_confidence(record)).attach(node)
            n_attached += 1

        if n_attached != len(nodes):
            logger.warning(
                "Transcribed %d of %d lines in '%s'; the rest failed line extraction.",
                n_attached, len(nodes), document.image_name,
            )
        return document


def _collect_lines(document: Document):
    """
    Walk the tree and describe every line in page coordinates.

    HTRflow stores a child polygon relative to its parent's bounding box, so
    the region origin has to be added back before handing anything to kraken.

    Returns:
        A `(nodes, lines)` pair of parallel lists, where `lines` holds
        `kraken.containers.BaselineLine` records.
    """
    from kraken.containers import BaselineLine, BBoxLine

    nodes, lines = [], []
    for region in document.regions:
        origin = region.polygon.bbox.p1
        for node in region.regions:
            boundary = [(int(x), int(y)) for x, y in node.polygon.move(origin)]
            baseline = node.annotations.get("baseline")

            if not baseline or len(baseline) < 2:
                # Not a kraken segmentation. Synthesise a baseline from the
                # bounding box so the line can still be recognized.
                bbox = Polygon(boundary).bbox
                baseline = BBoxLine(id="_", bbox=bbox.xyxy).to_baseline().baseline

            line_id = node.annotations.get("kraken_id") or f"_{uuid.uuid4()}"
            nodes.append(node)
            lines.append(
                BaselineLine(
                    id=line_id,
                    baseline=[(int(x), int(y)) for x, y in baseline],
                    boundary=boundary,
                )
            )
    return nodes, lines


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
