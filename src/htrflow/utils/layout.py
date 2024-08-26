from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from htrflow.utils.geometry import Bbox


if TYPE_CHECKING:
    from htrflow.volume.volume import Collection


logger = logging.getLogger(__name__)


def estimate_printspace(image: np.ndarray, window: int = 150) -> Bbox:
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
            Defaults to 150.

    Returns:
        The estimated printspace as a bounding box. If no printspace is
        detected, a bbox that covers the entire page is returned.
    """
    image = image.copy()
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Floodfill the image from the top-left corner. This removes (or
    # reduces) the dark border around the scanned page, which sometimes
    # interferes with the next step.
    _, image, *_ = cv2.floodFill(image, None, (0, 0), (255, 255, 255))

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
        for i in range(window, len(levels) - window):
            if np.median(levels[i - window : i]) > gray > np.median(levels[i : i + window]):
                break

        for j in range(len(levels) - window, window, -1):
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


class RegionLocation(Enum):
    MARGIN_TOP = 0
    PRINTSPACE = 1
    MARGIN_BOTTOM = 2
    MARGIN_LEFT = 3
    MARGIN_RIGHT = 4


def get_region_location(printspace: Bbox, region: Bbox) -> RegionLocation:
    """Get location of `region` relative to `printspace`

    The region is considered to be marginalia if more than 50% of it
    it located outside the printspace.

    The side margins extends to the top and bottom of the page. If the
    region is located in a corner, it will be assigned to the left or
    right margin and not the top or bottom margin.

    Arguments:
        printspace: A bounding box representing the page's printspace.
        region: The input region.

    Returns:
        A RegionLocation describing the region's location. Will default
        to RegionLocation.PRINTSPACE if the location cannot be decided.
    """
    overlap = printspace.intersection(region)
    if overlap is not None and (overlap.area / region.area) > 0.5:
        return RegionLocation.PRINTSPACE
    if region.xmax >= printspace.xmax:
        return RegionLocation.MARGIN_RIGHT
    if region.xmin <= printspace.xmin:
        return RegionLocation.MARGIN_LEFT
    if region.ymin <= printspace.ymin:
        return RegionLocation.MARGIN_TOP
    if region.ymax >= printspace.ymax:
        return RegionLocation.MARGIN_BOTTOM
    return RegionLocation.PRINTSPACE


def label_regions(collection: Collection):
    """Label collection's regions

    Labels each top-level segment of the collection as one of the five
    region types specified by geometry.RegionLocation. Saves the label
    in the node's data dictionary under `key`.

    Arguments:
        collection: Input collection
        key: Key used to save the region label. Defaults to
            "region_location".
    """

    for page in collection:
        printspace = estimate_printspace(page.image)
        for node in page:
            node.add_data(**{REGION_KEY: get_region_location(printspace, node.bbox)})


REGION_KEY = "region_location"
