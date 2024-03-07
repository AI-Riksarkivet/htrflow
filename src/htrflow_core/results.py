from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from htrflow_core import image


@dataclass
class Segment:
    """Segment class

    Attributes:
        bbox: The segment's bounding box as a tuple of coordinates (x1, x2, y1, y2)
        mask: The segment's mask, if available. The mask is relative to the bounding box.
        score: Segment confidence score
        class_label: Segment label, if available
        polygon: An approximation of the segment mask, !relative to the parent!
    """

    bbox: tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    polygon: list[tuple] = None
    score: Optional[float] = None
    class_label: Optional[str] = None
    # baseline: list[tuple] ?

    @classmethod
    def from_bbox(cls, bbox, **kwargs):
        """Create a segment from a bounding box"""
        mask = None
        polygon = image.bbox2polygon(bbox)
        return cls(bbox, mask, polygon, **kwargs)

    @classmethod
    def from_mask(cls, mask, **kwargs):
        """Create a segment from a mask

        Args:
            mask: A binary mask, of same shape as original image.
        """
        bbox = image.mask2bbox(mask)
        polygon = image.mask2polygon(mask)
        cropped_mask = image.crop(mask, bbox)
        return cls(bbox, cropped_mask, polygon, **kwargs)

    @classmethod
    def from_baseline(cls, baseline, **kwargs):
        """Create a segment from a baseline"""
        raise NotImplementedError()


@dataclass
class RecognizedText:
    """Recognized text class

    This class represents a result from a text recognition model.

    Attributes:
        texts: A sequence of candidate texts
        scores: The scores of the candidate texts
    """
    texts: Sequence[str]
    scores: Sequence[float]

    def top_candidate(self):
        """The best candidate text"""
        return self.texts[self.scores.index(self.top_score())]

    def top_score(self):
        """The highest score"""
        return max(self.scores)


@dataclass
class Result:
    """Result class

    This class bundles segmentation and text recognition results

    Returns:
        image: The original imaage
        metadata: Metadata associated with the result
        segments: Segments (may be empty)
        texts: Texts (may be empty)
    """

    image: np.ndarray
    metadata: dict
    segments: Sequence[Segment] = field(default_factory=list)
    texts: Sequence[RecognizedText] = field(default_factory=list)

    @classmethod
    def text_recognition_result(cls, image: np.ndarray, metadata: dict, text: RecognizedText) -> "Result":
        """Create a text recognition result

        Arguments:
            image: The original image
            metadata: Result metadata
            text: The recognized text

        Returns:
            A Result instance with the specified data and no segments.
        """
        return cls(image, metadata, texts=[text])

    @classmethod
    def segmentation_result(cls, image: np.ndarray, metadata: dict, segments: Sequence[Segment]) -> "Result":
        """Create a segmentation result

        Arguments:
            image: The original image
            metadata: Result metadata
            segments: The segments

        Returns:
            A Result instance with the specified data and no texts.
        """
        return cls(image, metadata, segments=segments)
