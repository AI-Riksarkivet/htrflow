from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

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

    masks_by_class: Dict[str, Sequence[Mask]] = defaultdict(list)
    for segment in result.segments:
        masks_by_class[segment.class_label].append(segment.global_mask)

    remove_indices_global = []

    for class_label, masks in masks_by_class.items():
        remove_indices = mask_nms(masks, containments_threshold)

        global_indices = [i for i, segment in enumerate(result.segments) if segment.class_label == class_label]
        remove_indices_global.extend([global_indices[i] for i in remove_indices])

    return remove_indices_global


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

    to_remove = np.any(significantly_contained & is_smaller_than_others, axis=1)
    remove_indices = np.where(to_remove)[0]

    return remove_indices.tolist()


def _calculate_area_comparison_matrix(stacked_masks: NDArray[np.uint8]) -> NDArray[np.bool_]:
    mask_areas = stacked_masks.sum(axis=(1, 2))
    mask_areas_expanded = mask_areas[:, np.newaxis]
    return mask_areas_expanded < mask_areas


def _calculate_containment_score(stacked_masks: NDArray[np.uint8], mask_i: Mask) -> NDArray[np.float_]:
    mask_i_expanded = np.expand_dims(mask_i, 0)
    intersections = np.logical_and(stacked_masks, mask_i_expanded).sum(axis=(1, 2)).astype(float)
    mask_a_area = mask_i_expanded.sum().astype(float)
    return intersections / mask_a_area if mask_a_area > 0 else 0
