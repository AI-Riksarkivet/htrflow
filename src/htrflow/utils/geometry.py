"""
Geometry utilities
"""

import logging
from dataclasses import astuple, dataclass
from typing import Iterable, Iterator, Sequence, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

Mask: TypeAlias = npt.NDArray[np.uint8]


@dataclass
class Point:
    """Class representing a point

    This class supports tuple-like unpacking and indexing.

    Attributes:
        x: The x-coordinate of the point
        y: The y-coordinate of the point

    Example:
    ```
    >>> p = Point(10, 20)
    >>> x, y = p
    >>> x == p[0]
    True
    ```
    """

    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        # Enables tuple-like iteration and unpacking
        return iter(astuple(self))

    def __getitem__(self, i: int) -> int:
        # Enables tuple-like indexing
        return (self.x, self.y)[i]

    def move(self, dest: "Point | tuple[int, int]") -> "Point":
        """Move point to `dest`

        Arguments:
            dest: A (dx, dy) tuple or Point specifying where to move.

        Returns:
            A copy of this Point instance moved `dx` along the x-axis
            and `dy` along the y-axis.

        Example:
        ```
        >>> Point(1, 2).move((10, 10))
        Point(11, 12)
        ```
        """
        dx, dy = dest
        return Point(self.x + dx, self.y + dy)

    def rescale(self, factor: float):
        """Rescale point by multiplying its coordinates with `factor`"""
        return Point(int(self.x * factor), int(self.y * factor))


@dataclass
class Bbox:
    """Bounding box class

    A dataclass that represents a bounding box. It supports tuple-like
    unpacking and indexing. For example:
    ```python
    >>> bbox = Bbox(0, 0, 10, 10)
    >>> xmin, ymin, xmax, ymax = bbox
    >>> ymax == bbox[3]
    True
    ```
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def __post_init__(self):
        # Make sure coordinates are integers
        self.xmin = int(self.xmin)
        self.xmax = int(self.xmax)
        self.ymin = int(self.ymin)
        self.ymax = int(self.ymax)

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
    def xyxy(self) -> tuple[int, int, int, int]:
        """Bounding box as a (xmin, ymin, xmax, ymax) tuple"""
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def xxyy(self) -> tuple[int, int, int, int]:
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
        """Center of bounding box rounded down to closest integer"""
        return Point(int((self.xmax + self.xmin) / 2), int((self.ymax + self.ymin) / 2))

    @property
    def area(self) -> int:
        """Area of bounding box"""
        return self.height * self.width

    def rescale(self, factor: float) -> "Bbox":
        """Rescale bounding box by multiplying its coordinates with `factor`"""
        return Bbox(*(int(coord * factor) for coord in self))

    def polygon(self) -> "Polygon":
        """Return a polygon representation of the bounding box"""
        return Polygon(
            [
                Point(self.xmin, self.ymin),
                Point(self.xmax, self.ymin),
                Point(self.xmax, self.ymax),
                Point(self.xmin, self.ymax),
            ]
        )

    def move(self, dest: Point | tuple[int, int]) -> "Bbox":
        """Move bounding box to `dest`

        Arguments:
            dest: A (dx, dy) tuple or Point.

        Returns:
            A copy of the bounding box with its coordinates shifted
            `dx` and `dy` in the x- and y-axis, respectively.
        """
        dx, dy = dest
        return Bbox(self.xmin + dx, self.ymin + dy, self.xmax + dx, self.ymax + dy)

    def intersects(self, other: "Bbox") -> bool:
        """Check if two bounding boxes intersect."""
        return not (
            self.xmax < other.xmin or self.xmin > other.xmax or self.ymax < other.ymin or self.ymin > other.ymax
        )

    def intersection(self, other: "Bbox") -> "Bbox | None":
        """Return the intersection between this bbox and `other`."""
        if not self.intersects(other):
            return None
        return Bbox(
            max(self.xmin, other.xmin),
            max(self.ymin, other.ymin),
            min(self.xmax, other.xmax),
            min(self.ymax, other.ymax),
        )

    def __iter__(self) -> Iterator[int]:
        # Enables tuple-like iteration and unpacking
        return iter(self.xyxy)

    def __getitem__(self, i: int) -> int:
        # Enables tuple-like indexing
        return self.xyxy[i]


class Polygon:
    """Polygon class

    This class represents a polygon as a sequence of `Point` instances.
    """

    points: Sequence[Point]

    def __init__(self, points: Iterable[tuple[int, int] | Point]):
        """Create a Polygon

        Attributes:
            points: The points defining the polygon, as either tuples
                or `Point` instances.
        """
        self.points = [Point(*point) for point in points]

    def move(self, dest: tuple[int, int] | Point) -> "Polygon":
        """Move polygon to `dest`

        Arguments:
            dest: A (dx, dy) tuple or Point.

        Returns:
            A copy of the polygon with its coordinates shifted
            `dx` and `dy` in the x- and y-axis, respectively.
        """
        return Polygon(point.move(dest) for point in self)

    def bbox(self) -> Bbox:
        """The smallest bounding box that encloses the polygon"""
        xs = [x for x, _ in self]
        ys = [y for _, y in self]
        return Bbox(min(xs), min(ys), max(xs), max(ys))

    def as_nparray(self) -> npt.NDArray[np.int32]:
        """The polygon as a [[x1, y1], ..., [xn, yn]] numpy array"""
        return np.array([[x, y] for x, y in self], dtype=np.int32)

    def rescale(self, factor: float) -> "Polygon":
        """Rescale polygon by multiplying its points with `factor`"""
        return Polygon([p.rescale(factor) for p in self])

    def __iter__(self) -> Iterator[Point]:
        return iter(self.points)

    def __getitem__(self, i: int) -> Point:
        return self.points[i]

    def __len__(self) -> int:
        return len(self.points)


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
    polygons = []
    for contour in contours:
        # `contour_epsilon` is the tolerance parameter `epsilon` adjusted
        # relative to the size of the mask
        contour_epsilon = epsilon * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, contour_epsilon, closed=True)
        squeezed = np.squeeze(approx)
        if squeezed.ndim == 1:
            continue
        polygons.append(Polygon(squeezed.tolist()))

    if len(polygons) > 1:
        logger.warning("Mask is not connected. Using the largest connected component")
        polygons.sort(key=lambda pol: pol.bbox().area, reverse=True)

    return polygons[0]


def masks2polygons(masks: Iterable[Mask], epsilon=0.005) -> list[Polygon]:
    """Convert masks to polygons"""
    return [mask2polygon(mask, epsilon) for mask in masks]


def mask2bbox(mask: Mask) -> Bbox:
    """Convert mask to bounding box"""
    y, x = np.where(mask != 0)
    return Bbox(np.min(x).item(), np.min(y).item(), np.max(x).item() + 1, np.max(y).item() + 1)


def bbox2mask(bbox: Bbox, shape: tuple[int, int]) -> Mask:
    """Create a mask from a bounding box

    Arguments:
        bbox: Intput bounding box
        shape: Shape of the desired mask as a (h, w) tuple
    """
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    return mask


def polygons2masks(mask: Mask, polygons: Iterable[Polygon]) -> list[Mask]:
    mask_height, mask_width = mask.shape[:2]
    masks = []
    for point in polygons:
        polygon = np.round(point).astype(np.int32)
        temp_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [polygon], color=255)
        masks.append(temp_mask)
    return masks
