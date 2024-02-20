from dataclasses import dataclass

import image
import numpy as np


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
    mask: np.ndarray | None
    score: float
    class_label: str | None
    polygon: list[tuple]
    # baseline: list[tuple] ?

    @classmethod
    def from_bbox(cls, bbox, **kwargs):
        """Create a segment from a bounding box"""
        mask = None
        polygon = image.bbox2polygon(bbox)
        return cls(bbox, mask, polygon=polygon, **kwargs)

    @classmethod
    def from_mask(cls, mask, **kwargs):
        """Create a segment from a mask

        Args:
            mask: A binary mask, of same shape as original image.
        """
        bbox = image.mask2bbox(mask)
        polygon = image.mask2polygon(mask)
        cropped_mask = image.crop(mask, bbox)
        return cls(bbox, cropped_mask, polygon=polygon, **kwargs)

    @classmethod
    def from_baseline(cls, baseline, **kwargs):
        """Create a segment from a baseline"""
        raise NotImplementedError()


@dataclass
class SegmentationResult:
    image: np.ndarray
    segments: list[Segment]

    def bboxes(self):
        return [segment.bbox for segment in self.segments]


@dataclass
class RecognitionResult:
    texts: list[str]
    scores: list[float]

    def top_candidate(self):
        return self.texts[self.scores.index(self.top_score())]

    def top_score(self):
        return max(self.score)
