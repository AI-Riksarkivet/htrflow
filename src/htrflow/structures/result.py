# from torch import Tensor
from typing import List

from htrflow.structures.seg_result import SegResult

# import numpy as np
# import cv2
# import torch
# from htrflow.utils.helper import timing_decorator
from htrflow.structures.text_rec_result import TextRecResult
import pandas as pd
import numpy as np
import torch


class Result():

    def __init__(self,
                img_shape,
                segmentation: SegResult,
                nested_results: List['Result'] = None,
                parent_result: 'Result' = None,
                texts: List['TextRecResult'] = None):

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
        sort_regions = lambda row: (
            row["page"],
            row["is_margin"],
            row["centroid_y"],
            row["centroid_x"],
        )

        # Sort the DataFrame using the custom function
        df["sort_key"] = df.apply(sort_regions, axis=1)
        df = df.sort_values("sort_key")

        # Return the ordered regions
        indices = df["region_id"].tolist()

        self._rearrange_instance(indices)

    def crop_regions_within_img(self, img):
        #continue with this one
        pass

    def order_instances(self):
        pass

    def convert_masks_to_polygons(self):
        pass

    def convert_res_to_page_xml(self):
        pass

"""
class FilterSegMask:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Removes smaller masks that are contained in a bigger mask
    # @timer_func
    def remove_overlapping_masks(self, predicted_mask, method="mask", containments_threshold=0.5):
        # Convert masks to binary images
        masks = [mask.cpu().numpy() for mask in predicted_mask.pred_instances.masks]
        masks_binary = [(mask > 0).astype(np.uint8) for mask in masks]

        masks_tensor = predicted_mask.pred_instances.masks
        masks_tensor = [mask.to(self.device) for mask in masks_tensor]

        # Compute bounding boxes and areas
        boxes = [cv2.boundingRect(mask) for mask in masks_binary]

        # Compute pairwise containment
        containments = np.zeros((len(masks), len(masks)))

        for i in range(len(masks)):
            box_i = boxes[i]

            for j in range(i + 1, len(masks)):
                box_j = boxes[j]

                if method == "mask":
                    containment = self._calculate_containment_mask(masks_tensor[i], masks_tensor[j])
                    containments[i, j] = containment
                    containment = self._calculate_containment_mask(masks_tensor[j], masks_tensor[i])
                    containments[j, i] = containment
                elif method == "bbox":
                    containment = self._calculate_containment_bbox(box_i, box_j)
                    containments[i, j] = containment
                    containment = self._calculate_containment_bbox(box_j, box_i)
                    containments[j, i] = containment

        # Keep only the biggest masks for overlapping pairs
        keep_mask = np.ones(len(masks), dtype=np.bool_)
        for i in range(len(masks)):
            if not keep_mask[i]:
                continue
            if np.any(containments[i] > containments_threshold):
                contained_indices = np.where(containments[i] > containments_threshold)[0]
                for j in contained_indices:
                    if np.count_nonzero(masks_binary[i]) >= np.count_nonzero(masks_binary[j]):
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False

        # Create a new DetDataSample with only selected instances
        filtered_result = DetDataSample(metainfo=predicted_mask.metainfo)
        pred_instances = InstanceData(metainfo=predicted_mask.metainfo)

        masks = [mask for i, mask in enumerate(masks) if keep_mask[i]]
        list_of_tensor_masks = [torch.from_numpy(mask) for mask in masks]
        stacked_masks = torch.stack(list_of_tensor_masks)

        updated_filtered_result = self._stacked_masks_update_data_sample(
            filtered_result, stacked_masks, pred_instances, keep_mask, predicted_mask
        )

        return updated_filtered_result

    def _stacked_masks_update_data_sample(self, filtered_result, stacked_masks, pred_instances, keep_mask, result):
        pred_instances.masks = stacked_masks
        pred_instances.bboxes = self._update_datasample_cat(result.pred_instances.bboxes.tolist(), keep_mask)
        pred_instances.scores = self._update_datasample_cat(result.pred_instances.scores.tolist(), keep_mask)
        pred_instances.kernels = self._update_datasample_cat(result.pred_instances.kernels.tolist(), keep_mask)
        pred_instances.labels = self._update_datasample_cat(result.pred_instances.labels.tolist(), keep_mask)
        pred_instances.priors = self._update_datasample_cat(result.pred_instances.priors.tolist(), keep_mask)

        filtered_result.pred_instances = pred_instances

        return filtered_result

    def _calculate_containment_bbox(self, box_a, box_b):
        xA = max(box_a[0], box_b[0])  # max x0
        yA = max(box_a[1], box_b[1])  # max y0
        xB = min(box_a[0] + box_a[2], box_b[0] + box_b[2])  # min x1
        yB = min(box_a[1] + box_a[3], box_b[1] + box_b[3])  # min y1

        box_a_area = box_a[2] * box_a[3]
        box_b_area = box_b[2] * box_b[3]

        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        containment = intersection_area / box_a_area if box_a_area > 0 else 0
        return containment

    def _calculate_containment_mask(self, mask_a, mask_b):
        intersection = torch.logical_and(mask_a, mask_b).sum().float()
        containment = intersection / mask_b.sum().float() if mask_b.sum() > 0 else 0
        return containment

    def _update_datasample_cat(self, cat_list, keep_mask):
        cat_keep = [cat for i, cat in enumerate(cat_list) if keep_mask[i]]
        tensor_cat_keep = torch.tensor(cat_keep)
        return tensor_cat_keep

    # @timer_func
    def filter_on_pred_threshold(self, result_pred, pred_score_threshold=0.5):
        id_list = []
        for id, pred_score in enumerate(result_pred.pred_instances.scores):
            if pred_score > pred_score_threshold:
                id_list.append(id)

        # Create a new DetDataSample with only selected instances
        new_filtered_result = DetDataSample(metainfo=result_pred.metainfo)
        new_pred_instances = InstanceData(metainfo=result_pred.metainfo)

        new_pred_instances.masks = result_pred.pred_instances.masks[id_list]
        new_pred_instances.bboxes = result_pred.pred_instances.bboxes[id_list]
        new_pred_instances.scores = result_pred.pred_instances.scores[id_list]
        new_pred_instances.kernels = result_pred.pred_instances.kernels[id_list]
        new_pred_instances.labels = result_pred.pred_instances.labels[id_list]
        new_pred_instances.priors = result_pred.pred_instances.priors[id_list]

        new_filtered_result.pred_instances = new_pred_instances
        return new_filtered_result
"""
