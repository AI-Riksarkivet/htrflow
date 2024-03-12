import numpy as np
import pandas as pd

from htrflow_core.results import SegmentationResult


def order_segments_marginalia(
    result: SegmentationResult, margin_ratio=0.2, histogram_bins=50, histogram_dip_ratio=0.5
):
    # Adapted from htrflow_core/src/transformations/order_segments.py

    # Create a pandas DataFrame from the regions
    df = pd.DataFrame(result.bboxes(), columns=["x_min", "x_max", "y_min", "y_max"])

    # Calculate the centroids of the bounding boxes
    df["centroid_x"] = (df["x_min"] + df["x_max"]) / 2
    df["centroid_y"] = (df["y_min"] + df["y_max"]) / 2

    # Calculate a histogram of the x-coordinates of the centroids
    histogram, _ = np.histogram(df["centroid_x"], bins=histogram_bins)

    # Determine if there's a significant dip in the histogram, which would suggest a two-page layout
    is_two_pages = np.min(histogram) < np.max(histogram) * histogram_dip_ratio

    image_width = result.image.shape[1]

    if is_two_pages:
        # Determine which page each region is on
        page_width = int(image_width / 2)
        df["page"] = (df["centroid_x"] > page_width).astype(int)

        # Determine if the region is in the margin
        margin_width = page_width * margin_ratio
        df["is_margin"] = ((df["page"] == 0) & (df["centroid_x"] < margin_width)) | (
            (df["page"] == 1) & (df["centroid_x"] > image_width - margin_width)
        )
    else:
        df["page"] = 0
        df["is_margin"] = (df["centroid_x"] < image_width * margin_ratio) | (
            df["centroid_x"] > image_width - page_width * margin_ratio
        )

    df = df.sort_values(by=["page", "is_margin", "centroid_y", "centroid_x"])

    # Reorder segments
    result.segments = [result.segments[i] for i in df.index.tolist()]


def order_lines(result: SegmentationResult, line_spacing_factor=0.5):
    # Adapted from htrflow_core/src/transformations/order_segments.py

    centroid_x = [(x1 + x2) / 2 for x1, x2, *_ in result.bboxes()]
    centroid_y = [(y1 + y2) / 2 for *_, y1, y2 in result.bboxes()]

    # Calculate the threshold distance
    average_line_height = np.mean([y2 - y1 for *_, y1, y2 in result.bboxes()])
    threshold_distance = average_line_height * line_spacing_factor

    # Sort the indices based on vertical center points and horizontal positions
    index = list(range(len(result.segments)))
    index.sort(key=lambda i: (centroid_x[i] // threshold_distance, centroid_y[i]))

    # Reorder segments
    result.segments = [result.segments[i] for i in index]
