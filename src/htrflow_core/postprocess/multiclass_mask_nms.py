from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from htrflow_core.logging.line_profiler import profile_performance
from htrflow_core.results import Result
from htrflow_core.utils.geometry import Mask


def multiclass_mask_nms(result: Result, containments_threshold: float = 0.5) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on masks across multiple classes based on containment scores.

    In gernermal, NMS is a post-processing technique used in object detection to eliminate duplicate detections
    and select the most relevant detected objects and this case the most relevant mask.

    This function segregates masks by their class labels and applies NMS within each class.
    Masks that are significantly ontained within larger masks,
    as defined by the containment threshold, are marked for removal.

    Args:
        result (Result): A Result object containing a sequence of masks and their associated class labels.
        containments_threshold (float): The threshold for deciding significant containment.

    Returns:
        List[int]: Indices of masks that should be removed after applying NMS.
    """

    remove_indices_global = []

    if len(result.segments) < 2:
        return remove_indices_global

    masks_by_class: Dict[str, Sequence[Mask]] = defaultdict(list)
    for segment in result.segments:
        masks_by_class[segment.class_label].append(segment.global_mask)

    for class_label, masks in masks_by_class.items():
        remove_indices = mask_nms(masks, containments_threshold)

        global_indices = [i for i, segment in enumerate(result.segments) if segment.class_label == class_label]
        remove_indices_global.extend([global_indices[i] for i in remove_indices])

    return remove_indices_global


@profile_performance
def mask_nms(masks: Sequence[Mask], containments_threshold: float = 0.5) -> List[int]:
    """
    Identify masks that should be removed based on containment scores and area comparisons.

    Args:
        masks (Sequence[Mask]): A sequence of masks to evaluate.
        containments_threshold (float): The threshold above which a mask is considered to be contained by another.

    Returns:
        List[int]: Indices of masks to be removed.
    """

    stacked_masks = np.stack(masks, axis=0)

    containments_score = np.array([_calculate_containment_score(stacked_masks, mask) for mask in stacked_masks])

    np.fill_diagonal(containments_score, 0)

    significantly_contained = containments_score > containments_threshold
    is_smaller_than_others = _calculate_area_comparison_matrix(stacked_masks)

    print(is_smaller_than_others)

    to_remove = np.any(significantly_contained & is_smaller_than_others, axis=1)

    print(to_remove)

    remove_indices = np.where(to_remove)[0]

    return remove_indices.tolist()


def _calculate_area_comparison_matrix(stacked_masks):
    mask_areas = stacked_masks.sum(axis=(1, 2))
    mask_areas_expanded = mask_areas[:, np.newaxis]
    return mask_areas_expanded < mask_areas


def _calculate_containment_score(stacked_masks, mask_i: Mask):
    mask_i_expanded = np.expand_dims(mask_i, 0)
    intersections = np.logical_and(stacked_masks, mask_i_expanded).sum(axis=(1, 2)).astype(float)
    mask_a_area = mask_i_expanded.sum().astype(float)
    return intersections / mask_a_area if mask_a_area > 0 else 0


if __name__ == "__main__":
    from htrflow_core.results import Result

    def results_with_mask():
        orig_shape = (200, 200)
        image = np.zeros(orig_shape, dtype=np.uint8)

        mask_a = np.zeros(orig_shape, dtype=np.uint8)
        mask_a[20:70, 20:70] = 1  # Small square mask
        mask_b = np.zeros(orig_shape, dtype=np.uint8)
        mask_b[10:180, 10:180] = 1  # Large square mask
        mask_c = np.zeros(orig_shape, dtype=np.uint8)
        mask_c[30:70, 30:70] = 1  # Overlapping small square mask
        mask_d = np.zeros(orig_shape, dtype=np.uint8)
        mask_d[50:100, 50:100] = 1  # Another overlapping mask in class_1
        mask_e = np.zeros(orig_shape, dtype=np.uint8)
        mask_e[120:160, 120:160] = 1  # Non-overlapping mask in class_2

        return [mask_a, mask_c, mask_d]

        # segment_a = Segment(mask=mask_a, class_label="class_1", orig_shape=orig_shape)
        # # segment_b = Segment(mask=mask_b, class_label="class_2", orig_shape=orig_shape)
        # segment_c = Segment(mask=mask_c, class_label="class_1", orig_shape=orig_shape)
        # segment_d = Segment(mask=mask_d, class_label="class_1", orig_shape=orig_shape)
        # # segment_e = Segment(mask=mask_e, class_label="class_3", orig_shape=orig_shape)

        # return Result(image=image, metadata={}, segments=[segment_a, segment_c, segment_d])

    result = results_with_mask()

    drop_id = mask_nms(result)

    print(drop_id)

    # result.drop_indices(drop_id)

    # helper_plot_for_segment(result.segments, result.image)
