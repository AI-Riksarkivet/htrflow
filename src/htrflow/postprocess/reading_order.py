from typing import Sequence

from htrflow.utils.geometry import Bbox
from htrflow.utils.layout import get_region_location
from htrflow.volume.volume import ImageNode


def order_regions(regions: Sequence[ImageNode], printspace: Bbox, is_twopage: bool = False):
    """Order regions according to their reading order

    This function estimates the reading order based on the following:
        1. Which page of the spread the region belongs to (if
            `is_twopage` is True)
        2. Where the region is located relative to the page's
            printspace. The ordering is: top margin, printspace,
            bottom margin, left margin, right margin. See
            `layout.RegionLocation` for more details.
        3. The y-coordinate of the region's top-left corner.

    This function can be used to order the top-level regions of a
    page, but is also suitable for ordering the lines within each
    region.

    Arguments:
        regions: Regions to be ordered.
        printspace: A bounding box around the page's printspace.
        is_twopage: Whether the page is a two-page spread.

    Returns:
        The input regions in reading order.
    """
    index = order_bboxes([region.bbox for region in regions], printspace, is_twopage)
    return [regions[i] for i in index]


def order_bboxes(bboxes: Sequence[Bbox], printspace: Bbox, is_twopage: bool):
    """Order bounding boxes with respect to printspace

    This function estimates the reading order based on the following:
        1. Which page of the spread the bounding box belongs to (if
            `is_twopage` is True)
        2. Where the bounding box is located relative to the page's
            printspace. The ordering is: top margin, printspace,
            bottom margin, left margin, right margin. See
            `layout.RegionLocation` for more details.
        3. The y-coordinate of the bounding box's top-left corner.

    Arguments:
        bboxes: Bounding boxes to be ordered.
        printspace: A bounding box around the page's printspace.
        is_twopage: Whether the page is a two-page spread.

    Returns:
        A list of integers `index` where `index[i]` is the suggested
        reading order of the i:th bounding box.
    """

    def key(i: int):
        return (
            is_twopage and (bboxes[i].center.x > printspace.center.x),
            get_region_location(printspace, bboxes[i]).value,
            bboxes[i].ymin,
        )

    return sorted(range(len(bboxes)), key=key)


def left_right_top_down(bboxes: Sequence[Bbox], line_spacing: float | None = 1.0):
    """Order bounding boxes left-right top-down

    This function orders the input boxes after their top left corner,
    i.e., their minimum x and y coordinates. The y coordinates are
    discretized based on the `line_spacing` parameter.

    Arguments:
        bboxes: Input bounding boxes
        line_spacing: A parameter that controls the discretization of
            the y coordinates. A higher value will move the boxes more
            along the y axis before ordering. If None, the boxes will
            not be moved at all.

    Returns:
        A sorted index.
    """
    xs = [bbox.xmin for bbox in bboxes]
    ys = [bbox.ymin for bbox in bboxes]

    if line_spacing and bboxes:  # `and bboxes` to avoid division by zero
        average_height = sum(bbox.height for bbox in bboxes) / len(bboxes)
        threshold = average_height * line_spacing
        ys = [y // threshold for y in ys]

    return sorted(range(len(bboxes)), key=lambda i: (ys[i], xs[i]))


def top_down(bboxes: Sequence[Bbox]):
    """Order bounding boxes top-down"""
    return sorted(range(len(bboxes)), key=lambda i: bboxes[i].center.y)
