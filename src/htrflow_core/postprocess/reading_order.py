from typing import Sequence

import numpy as np

from htrflow_core.results import Result
from htrflow_core.utils.geometry import Bbox


def is_margin(region: Bbox, page: Bbox, margin_ratio: float=0.2):
    """Check if `region` lies in the (left or right) margin of `page`

    The region is assumed to be marginalia if its center point on the
    x-axis lies close to the left or right side of the page.

    Arguments:
        region: The region to check
        page: The boundaries of the page that the region lies in
        margin_ratio: Margin to page width ratio. Default is 0.2.
    """
    margin = page.width * margin_ratio
    left_margin = page.x1 + margin
    right_margin = page.x2 - margin
    return (region.center.x < left_margin or region.center.x > right_margin)


def is_twopage(segments: Sequence[Bbox], histogram_bins=50, histogram_dip_ratio=0.5):
    xs = [segment.center.x for segment in segments]

    # Calculate a histogram of the x-coordinates of the centroids
    histogram, _ = np.histogram(xs, bins=histogram_bins)

    # Determine if there's a significant dip in the histogram, which would suggest a two-page layout
    return np.min(histogram) < np.max(histogram) * histogram_dip_ratio


def order_segments_marginalia(result: Result, histogram_bins=50, histogram_dip_ratio=0.5):
    # Adapted from htrflow_core/src/transformations/order_segments.py

    xs = [segment.center.x for segment in result.segments]
    ys = [segment.center.y for segment in result.segments]

    image_height, image_width = result.image.shape[:2]
    index = list(range(len(result.segments)))

    if is_twopage(result.segments, histogram_bins, histogram_dip_ratio):
        page_width = int(image_width / 2)
        pagei = [x < page_width for x in xs]
        pages = [Bbox(0, page_width, 0, image_height), Bbox(page_width, image_width, 0, image_height)]
        index.sort(key=lambda i: (pagei[i], is_margin(result.segments[i], pages[pagei[i]], ys[i], xs[i])))
    else:
        page = Bbox(0, image_width, 0, image_height)
        index.sort(key=lambda i: (is_margin(result.segments[i], page), ys[i], xs[i]))

    return index


def order_lines(result: Result, line_spacing_factor=0.5) -> Sequence[int]:
    # Adapted from htrflow_core/src/transformations/order_segments.py

    xs = [segment.center.x for segment in result.segments]
    ys = [segment.center.y for segment in result.segments]

    # Calculate the threshold distance
    average_line_height = sum(segment.height for segment in result.segments) / len(result.segments)
    threshold_distance = average_line_height * line_spacing_factor

    # Sort the indices based on vertical center points and horizontal positions
    index = list(range(len(result.segments)))
    index.sort(key=lambda i: (xs[i] // threshold_distance, ys[i]))
    return index
