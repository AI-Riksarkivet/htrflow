"""
Geometry utilities
"""

from collections import namedtuple
from dataclasses import astuple, dataclass
from typing import Iterable, Sequence, TypeAlias

import cv2
import numpy as np


Point = namedtuple("Point", ["x", "y"])
Polygon: TypeAlias = Sequence[Point] | Sequence[tuple[int, int]]
Mask: TypeAlias = np.ndarray[np.uint8]


@dataclass
class Bbox:
    """Bounding box class"""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def height(self) -> int:
        """Height of bounding box"""
        return self.ymax - self.ymin

    @property
    def width(self) -> int:
        """Width of bounding box"""
        return self.xmax - self.xmin

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        """Bounding box as a (xmin, ymin, width, height) tuple"""
        return self.xmin, self.ymin, self.width, self.height

    @property
    def xyxy(self):
        """Bounding box as a (xmin, ymin, xmax, ymax) tuple"""
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def xxyy(self):
        """Bounding box as a (xmin, xmax, ymin, ymax) tuple"""
        return self.xmin, self.xmax, self.ymin, self.ymax

    @property
    def p1(self) -> Point:
        """Top left corner of bounding box (xmin, ymin)"""
        return Point(self.xmin, self.ymin)

    @property
    def p2(self) -> Point:
        """Bottom right corner of bounding box (xmax, ymax)"""
        return Point(self.xmax, self.ymax)

    @property
    def center(self) -> Point:
        """Center of bounding box"""
        return Point((self.xmax - self.xmin) / 2, (self.ymax - self.ymin) / 2)

    def __iter__(self):
        # Tuple-like iteration and unpacking
        return iter(astuple(self))

    def __getitem__(self, i):
        # Tuple-like indexing
        return astuple(self)[i]


def mask2polygon(mask: Mask, epsilon: float = 0.005) -> Polygon:
    """Convert mask to polygon

    Args:
        mask: The input mask
        epsilon: A tolerance parameter. Smaller epsilon will result in
            a higher-resolution polygon.

    Returns:
        A list of coordinate tuples
    """
    # Ensure mask is 8-bit single-channel image
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) == 3 and mask.shape[2] != 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # `contours` is a high-resolution contour, but we need a simple polygon, so
    # we approximate it with the DP algorithm, which removes as many points as
    # possible while still keeping the original shape.

    # Adjust the tolerance parameter `epsilon` relative to the size of the mask
    epsilon *= cv2.arcLength(contours[0], closed=True)
    approx = cv2.approxPolyDP(contours[0], epsilon, closed=True)
    return np.squeeze(approx)


def masks2polygons(masks: Iterable[Mask], epsilon=0.005) -> Iterable[Polygon]:
    """Convert masks to polygons"""
    return [mask2polygon(mask, epsilon) for mask in masks]


def bbox2polygon(bbox: Bbox) -> Polygon:
    """Convert bounding box to polygon"""
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])


def mask2bbox(mask: Mask) -> Bbox:
    """Convert mask to bounding box"""
    y, x = np.where(mask != 0)
    return np.min(x), np.min(y), np.max(x), np.max(y)
