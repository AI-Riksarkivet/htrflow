from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from htrflow.results import Result
from htrflow.utils.geometry import Mask


def multiclass_mask_nms(result: Result, containments_threshold: float = 0.5, downscale: float = 0.25) -> List[int]:
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
        downscale (float): If < 1, NMS will be performed on lower resolution versions of the masks,
            downscaled to this factor. Example: downscale=0.5 means that the masks are halved in size
            (number of pixels). This speeds up the computation for larger masks, at the expense of NMS
            accuracy.

    Returns:
        List[int]: Indices of masks that should be removed after applying NMS.
    """
    if len(result.segments) < 2:
        return []

    downscale = min(downscale, 1)  # Set downscale factor to at most 1

    remove_indices_global = []

    masks_by_class: Dict[str, Sequence[Mask]] = defaultdict(list)
    for segment in result.segments:
        masks_by_class[segment.class_label].append(segment.approximate_mask(downscale))

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

    containments_score = calculate_containment_scores(stacked_masks)

    np.fill_diagonal(containments_score, 0)

    significantly_contained = containments_score > containments_threshold
    is_smaller_than_others = _calculate_area_comparison_matrix(stacked_masks)

    to_remove = np.any(significantly_contained & is_smaller_than_others, axis=1)

    remove_indices = np.where(to_remove)[0]

    return remove_indices.tolist()


def _calculate_area_comparison_matrix(stacked_masks):
    mask_areas = stacked_masks.sum(axis=(1, 2))
    mask_areas_expanded = mask_areas[:, np.newaxis]
    return mask_areas_expanded < mask_areas


def calculate_containment_scores(stacked_masks):
    """
    Calculate containment scores for all masks in a stacked array,
    shape (N, H, W) where N is the number of masks, and H and W are the dimensions of each mask
    Return is a 2D numpy array (N, N) containing the containment scores of each mask within every other mask.
    """

    # Calculate intersections: (N, 1, H, W) AND (1, N, H, W) => (N, N, H, W), then sum over H and W
    intersections = (
        np.logical_and(stacked_masks[:, np.newaxis], stacked_masks[np.newaxis, :]).sum(axis=(2, 3)).astype(float)
    )

    # Calculate areas of each mask, broadcasted to shape (N, N)
    areas = stacked_masks.sum(axis=(1, 2)).astype(float)

    # Avoid division by zero by ensuring no zero areas
    areas[areas == 0] = 1

    # each mask's intersection with another mask divided by its own area
    containment_scores = intersections / areas[:, np.newaxis]

    return containment_scores


if __name__ == "__main__":
    import random

    import numpy as np

    from htrflow.results import Result, Segment
    from htrflow.utils.draw import helper_plot_for_segment

    def generate_random_masks(num_masks, image_size=(200, 200), num_classes=2):
        random.seed(42)
        np.random.seed(42)
        masks = []
        classes = [f"class_{i+1}" for i in range(num_classes)]
        for _ in range(num_masks):
            w, h = random.randint(20, 100), random.randint(20, 40)  # Random width and height
            x, y = random.randint(0, image_size[0] - w), random.randint(0, image_size[1] - h)  # Random position
            mask = np.zeros(image_size, dtype=np.uint8)
            mask[y : y + h, x : x + w] = 1
            class_label = random.choice(classes)
            masks.append(Segment(mask=mask, class_label=class_label, orig_shape=image_size))
        return masks

    def simulate_large_dataset():
        num_masks = 100  # Simulating a large number of masks
        image = np.zeros((200, 200), dtype=np.uint8)
        segments = generate_random_masks(num_masks)
        return Result(image=image, metadata={}, segments=segments)

    result = simulate_large_dataset()

    helper_plot_for_segment(result.segments, result.image)

    drop_id = multiclass_mask_nms(result)

    print(drop_id)

    result.drop_indices(drop_id)

    helper_plot_for_segment(result.segments, result.image)
