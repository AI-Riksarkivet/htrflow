from dataclasses import dataclass
from typing import Optional

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
    def from_bbox(cls, bbox, *args):
        """Create a segment from a bounding box"""
        mask = None
        polygon = image.bbox2polygon(bbox)
        return cls(bbox, mask, polygon, *args)

    @classmethod
    def from_mask(cls, mask, *args):
        """Create a segment from a mask

        Args:
            mask: A binary mask, of same shape as original image.
        """
        bbox = image.mask2bbox(mask)
        polygon = image.mask2polygon(mask)
        cropped_mask = image.crop(mask, bbox)
        return cls(bbox, cropped_mask, polygon, **args)

    @classmethod
    def from_baseline(cls, baseline, *args):
        """Create a segment from a baseline"""
        raise NotImplementedError()


class Result:
    """Result base class"""
    metadata: dict

@dataclass
class SegmentationResult(Result):
    image: np.ndarray
    segments: list[Segment]

    def bboxes(self):
        return [segment.bbox for segment in self.segments]

    def polygons(self):
        return [segment.polygon for segment in self.segments]

    @classmethod
    def from_bboxes(cls, image, bboxes, *args):
        segments = [Segment.from_bbox(*item) for item in zip(bboxes, *args)]
        return cls(image, segments)

    @classmethod
    def from_masks(cls, image, masks, *args):
        segments = [Segment.from_mask(*item) for item in zip(masks, *args)]
        return cls(image, segments)

    def save(self, dest: str):
        img = image.draw_polygons(self.image, self.bboxes())
        image.write(dest, img)


@dataclass
class RecognitionResult(Result):
    texts: list[str]
    scores: list[float]

    def top_candidate(self):
        return self.texts[self.scores.index(self.top_score())]

    def top_score(self):
        return max(self.scores)