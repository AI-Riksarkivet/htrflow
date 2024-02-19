# from torch import Tensor
from typing import List

import numpy as np
import pandas as pd
import torch
from htrflow.structures.seg_result import SegResult

# import numpy as np
# import cv2
# import torch
# from htrflow.utils.helper import timing_decorator
from htrflow.structures.text_rec_result import TextRecResult


class Result:
    def __init__(
        self,
        img_shape,
        segmentation: SegResult,
        nested_results: List["Result"] = None,
        parent_result: "Result" = None,
        texts: List["TextRecResult"] = None,
    ):
        self.img_shape = img_shape
        self.segmentation = segmentation
        self.nested_results = nested_results
        self.parent_result = parent_result
        self.texts = texts

    def _rearrange_instance(self, indices):
        self.segmentation.masks = torch.stack([self.segmentation.masks[i] for i in indices])
        self.segmentation.labels = torch.stack([self.segmentation.labels[i] for i in indices])
        self.segmentation.scores = torch.stack([self.segmentation.scores[i] for i in indices])
        self.segmentation.bboxes = torch.stack([self.segmentation.masks[i] for i in indices])

    def order_lines(self, line_spacing_factor=0.5):
        bounding_boxes = self.segmentation.bboxes.tolist()
        center_points = [(box[1] + box[3]) / 2 for box in bounding_boxes]
        horizontal_positions = [(box[0] + box[2]) / 2 for box in bounding_boxes]

        # Calculate the threshold distance
        threshold_distance = self._calculate_threshold_distance(bounding_boxes, line_spacing_factor)

        # Sort the indices based on vertical center points and horizontal positions
        indices = list(range(len(bounding_boxes)))
        indices.sort(
            key=lambda i: (
                center_points[i] // threshold_distance,
                horizontal_positions[i],
            )
        )

        # Order text lines
        self._rearrange_instance(indices)

    def _calculate_threshold_distance(self, bounding_boxes, line_spacing_factor=0.5):
        # Calculate the average height of the text lines
        total_height = sum(box[3] - box[1] for box in bounding_boxes)
        average_height = total_height / len(bounding_boxes)

        # Calculate the threshold distance, Set a factor for the threshold distance (adjust as needed)
        threshold_distance = average_height * line_spacing_factor

        # Return the threshold distance
        return threshold_distance

    def order_regions_marginalia(self, region_image, margin_ratio=0.2, histogram_bins=50, histogram_dip_ratio=0.5):
        bounding_boxes = self.segmentation.bboxes.tolist()
        img_width = self.img_shape[1]

        regions = [[i, x[0], x[1], x[0] + x[2], x[1] + x[3]] for i, x in enumerate(bounding_boxes)]

        # Create a pandas DataFrame from the regions
        df = pd.DataFrame(regions, columns=["region_id", "x_min", "y_min", "x_max", "y_max"])

        # Calculate the centroids of the bounding boxes
        df["centroid_x"] = (df["x_min"] + df["x_max"]) / 2
        df["centroid_y"] = (df["y_min"] + df["y_max"]) / 2

        # Calculate a histogram of the x-coordinates of the centroids
        histogram, bin_edges = np.histogram(df["centroid_x"], bins=histogram_bins)

        # Determine if there's a significant dip in the histogram, which would suggest a two-page layout
        is_two_pages = np.min(histogram) < np.max(histogram) * histogram_dip_ratio

        if is_two_pages:
            # Determine which page each region is on
            page_width = int(img_width / 2)
            df["page"] = (df["centroid_x"] > page_width).astype(int)

            # Determine if the region is in the margin
            margin_width = page_width * margin_ratio
            df["is_margin"] = ((df["page"] == 0) & (df["centroid_x"] < margin_width)) | (
                (df["page"] == 1) & (df["centroid_x"] > img_width - margin_width)
            )
        else:
            df["page"] = 0
            df["is_margin"] = (df["centroid_x"] < img_width * margin_ratio) | (
                df["centroid_x"] > img_width - page_width * margin_ratio
            )

        # Define a custom sorting function
        def sort_regions(row):
            return row["page"], row["is_margin"], row["centroid_y"], row["centroid_x"]

        # Sort the DataFrame using the custom function
        df["sort_key"] = df.apply(sort_regions, axis=1)
        df = df.sort_values("sort_key")

        # Return the ordered regions
        indices = df["region_id"].tolist()

        self._rearrange_instance(indices)
