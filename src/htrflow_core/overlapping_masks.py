from typing import List, Sequence

import numpy as np

from htrflow_core.results import Result
from htrflow_core.utils.geometry import Mask


# Todo add class_label filtering


def multiclass_mask_nms(result: Result, containments_threshold: float = 0.5):
    return find_overlapping_masks_to_remove(result.global_masks, containments_threshold)


def find_overlapping_masks_to_remove(masks: Sequence[Mask], containments_threshold: float = 0.5) -> List[int]:
    stacked_masks = np.stack(masks, axis=0)

    containments_score = np.array([_calculate_containment_score(stacked_masks, mask) for mask in stacked_masks])
    np.fill_diagonal(containments_score, 0)

    significantly_contained = containments_score > containments_threshold
    is_smaller_than_others = _calculate_area_comparison_matrix(stacked_masks)

    to_remove = np.any(significantly_contained & is_smaller_than_others, axis=1)
    remove_indices = np.where(to_remove)[0]

    return remove_indices.tolist()


def _calculate_area_comparison_matrix(masks):
    mask_areas = masks.sum(axis=(1, 2))
    mask_areas_expanded = mask_areas[:, np.newaxis]
    return mask_areas_expanded < mask_areas


def _calculate_containment_score(masks, mask_i):
    mask_i_expanded = np.expand_dims(mask_i, 0)
    intersections = np.logical_and(masks, mask_i_expanded).sum(axis=(1, 2)).astype(float)
    mask_a_area = mask_i_expanded.sum().astype(float)
    return intersections / mask_a_area if mask_a_area > 0 else 0


if __name__ == "__main__":
    import cv2

    from htrflow_core.models.openmmlab.rmtdet import RTMDet
    from htrflow_core.utils.draw import helper_plot_for_segment

    model = RTMDet(
        model="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/model.pth",
        config="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/config.py",
        device="cuda:0",
    )

    img2 = "/home/gabriel/Desktop/htrflow_core/data/demo_images/demo_image.jpg"
    image2 = cv2.imread(img2)

    results = model([image2], pred_score_thr=0.4)

    index_to_drop = multiclass_mask_nms(results[0])

    results[0].drop(index_to_drop)

    print(results[0])

    helper_plot_for_segment(image2, results[0].segments, maskalpha=0.5, boxcolor=None)
