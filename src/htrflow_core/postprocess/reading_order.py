from typing import Sequence

from htrflow_core.utils.geometry import Bbox
from htrflow_core.utils.layout import get_region_location


def order_region_with_marginalia(printspace: Bbox, bboxes: Sequence[Bbox]):
    """Order bounding boxes with respect to printspace

    This function orders the input boxes after their location relative
    to the printspace (see layout.RegionLocation) and then their
    y-coordinate. It is useful whenever a region is undersegmented
    (there are several "regions" within the region)
    """
    locations = [get_region_location(printspace, bbox) for bbox in bboxes]
    return sorted(range(len(bboxes)), key=lambda i: (locations[i].value, bboxes[i].ymin))


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

    if line_spacing:
        average_height = sum(bbox.height for bbox in bboxes) / len(bboxes)
        threshold = average_height * line_spacing
        ys = [y // threshold for y in ys]

    return sorted(range(len(bboxes)), key=lambda i: (ys[i], xs[i]))
