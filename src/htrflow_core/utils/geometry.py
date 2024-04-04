"""
Geometry utilities
"""
import logging
from dataclasses import astuple, dataclass
from typing import Iterable, Sequence, TypeAlias

import cv2
import numpy as np


logger = logging.getLogger(__name__)

Mask: TypeAlias = np.ndarray[np.uint8]


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

    def __iter__(self) -> Iterable[int]:
        # Enables tuple-like iteration and unpacking
        return iter(astuple(self))

    def __getitem__(self, i: int) -> int:
        # Enables tuple-like indexing
        return astuple(self)[i]


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

    def polygon(self) -> "Polygon":
        """Return a polygon representation of the bounding box"""
        return Polygon([
            Point(self.xmin, self.ymin),
            Point(self.xmax, self.ymin),
            Point(self.xmax, self.ymax),
            Point(self.xmin, self.ymax),
        ])

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

    def __iter__(self):
        # Enables tuple-like iteration and unpacking
        return iter(astuple(self))

    def __getitem__(self, i: int) -> int:
        # Enables tuple-like indexing
        return astuple(self)[i]


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

    def as_nparray(self):   # -> n x 2 numpy array of integers
        """A np array version of the polygon"""
        return np.array([[x, y] for x, y in self])

    def __iter__(self):  # -> Iterable[Point]
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

    # Adjust the tolerance parameter `epsilon` relative to the size of the mask
    epsilon *= cv2.arcLength(contours[0], closed=True)
    approx = cv2.approxPolyDP(contours[0], epsilon, closed=True)
    return Polygon(np.squeeze(approx).tolist())


def masks2polygons(masks: Iterable[Mask], epsilon=0.005) -> Iterable[Polygon]:
    """Convert masks to polygons"""
    return [mask2polygon(mask, epsilon) for mask in masks]


def mask2bbox(mask: Mask) -> Bbox:
    """Convert mask to bounding box"""
    y, x = np.where(mask != 0)
    return Bbox(np.min(x).item(), np.min(y).item(), np.max(x).item(), np.max(y).item())


def estimate_printspace(image: np.ndarray, window: int = 50) -> Bbox:
    """Estimate printspace of page

    The printspace (borrowed terminology from ALTO XML) is a
    rectangular area that covers the main text body. Margins, page
    numbers and (in some cases) titles are not part of the printspace.

    This function estimates the printspace from the given image based
    on its pixel values. It works on pages with simple one- or two-
    page layouts with a moderate amount of marginalia. It only detects
    one printspace, even if the image has a two-page layout. If both
    printspaces need to be detected, the image needs to be cropped
    before this function is used.

    Args:
        image (np.ndarray): The input image as a numpy array, in
            grayscale or BGR.
        window (int, optional): A tolerance parameter. A large window
            makes the function less sensible to noise, but more prone
            to produce a result that does not cover the actual
            printspace entirely. A small window is more sensible to
            noise, and more prone to capture marignalia as printspace.
            Defaults to 50.

    Returns:
        The estimated printspace as a bounding box. If no printspace is
        detected, a bbox that covers the entire page is returned.
    """
    image = image.copy()
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Floodfill the image from the top-left corner. This removes (or
    # reduces) the dark border around the scanned page, which sometimes
    # interferes with the next step.
    _, image, *_ = cv2.floodFill(image, None, (0,0), (255, 255, 255))

    # The bounding box is produced in two steps: First the left-right
    # boundaries are found, then the top-bottom boundaries.
    bbox = [0, 0, 0, 0]
    for axis in (0, 1):
        # Create a vector `levels` that represents the ratio of black
        # to white pixels along the axis.
        levels = image.sum(axis=axis).astype(np.float64)
        levels /= np.max(levels)

        # Find the average gray value by taking the mean of `levels`,
        # excluding the 10% lightest and 10% darkest rows/columns.
        levels_sorted = np.sort(levels)
        a = 0.1
        mids = levels_sorted[int(len(levels) * a) : int((1 - a) * len(levels))]
        gray = np.mean(mids)

        # Find the first point where the lightness drops below `gray`, and
        # stays rather stable below it. The intuition here is that the
        # printspace is generally darker than the average gray point.
        # Instead of taking the actual values at row/colum i, the median
        # values over a range ahead is compared with the median value of
        for i in range(window, len(levels)-window):
            if np.median(levels[i - window : i]) > gray > np.median(levels[i : i + window]):
                break

        for j in range(len(levels)-window, window, -1):
            if np.median(levels[j - window : j]) < gray < np.median(levels[j : j + window]):
                break

        if i > j:
            i = 0
            j = image.shape[1 - axis]
            logger.warning(f"Could not find printspace along axis {axis}.")

        bbox[axis] = i
        bbox[axis + 2] = j

    return Bbox(*bbox)


def is_twopage(img, strip_width=0.1, threshold=0.2):
    """Detect if image deptics a two-page spread

    This function detects a dark vertical line within a strip in the
    middle of the image. More specifically, it checks if the darkest
    column of pixels within the middle strip is among the darkest 10%
    columns of the entire image.

    This function will not detect two-page documents without a dark
    divider between the two pages.

    Args:
        image: Input image in grayscale or BGR.
        strip_width: Width of the strip to check for dark lines,
            relative to the image width. Defaults to 0.1, i.e., the
            middle 10% of the image will be checked.
        threshold: Detection threshold, range [0, 1], recommended range
            about (0.1, 0.4). A higher value is more prone to false
            positives whereas a lower value is more prone to false
            negatives.

    Returns:
       The location (y-coordinate in matrix notation) of the detected
       divider, if found, else None.
    """
    img = img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w = img.shape[1]
    middle = int(w / 2)
    half_strip = int(strip_width * w / 2)
    levels = img.sum(axis=0)
    strip = levels[middle - half_strip : middle + half_strip]

    # Check if min value of strip is among the darkest `threshold` %
    # of the image. If no dark divider is present, the minimum value of
    # the strip should be closer to the median, i.e., around 50%.
    if np.min(strip) < np.sort(levels)[int(w * threshold)]:
        return middle - half_strip + np.argmin(strip)
    return None


class RegionLocation:
    PRINTSPACE = "printspace"
    MARGIN_LEFT = "margin_left"
    MARGIN_RIGHT = "margin_right"
    MARGIN_TOP = "margin_top"
    MARGIN_BOTTOM = "margin_bottom"


def get_region_location(printspace: Bbox, region: Bbox) -> RegionLocation:
    """Get location of `region` relative to `printspace`

    The side margins extends to the top and bottom of the page. If the
    region is located in a corner, it will be assigned to the left or
    right margin and not the top or bottom margin.
    """
    if region.center.x < printspace.xmin:
        return RegionLocation.MARGIN_LEFT
    elif region.center.x > printspace.xmax:
        return RegionLocation.MARGIN_RIGHT
    elif region.center.y > printspace.ymax:
        return RegionLocation.MARGIN_TOP
    elif region.center.y < printspace.ymin:
        return RegionLocation.MARGIN_BOTTOM
    return RegionLocation.PRINTSPACE
