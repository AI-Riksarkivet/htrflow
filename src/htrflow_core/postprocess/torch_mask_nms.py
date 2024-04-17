from typing import List

import numpy as np
import torch

from htrflow_core.logging.line_profiler import profile_performance


@profile_performance
def torch_mask_nms(masks: torch.Tensor, containments_threshold: float = 0.5) -> List[int]:
    """
    Identify masks that should be removed based on containment scores and area comparisons.

    Args:
        masks (torch.Tensor): A 3D Tensors [b,H,W].
        containments_threshold (float): The threshold above which a mask is considered to be contained by another.

    Returns:
        List[int]: Indices of masks to be removed.
    """

    mask_areas = masks.sum(dim=(1, 2))

    # expanded_masks = masks.unsqueeze(0)  # Shape [1, N, H, W]
    # transposed_masks = masks.unsqueeze(1)  # Shape [N, 1, H, W]
    # intersections = (expanded_masks & transposed_masks).sum(dim=(2, 3))  # Shape [N, N]

    intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).sum(dim=(2, 3))

    containments_score = intersections / mask_areas.unsqueeze(1)

    torch.diagonal(containments_score).fill_(0)

    significantly_contained = containments_score > containments_threshold
    is_smaller_than_others = mask_areas.unsqueeze(0) > mask_areas.unsqueeze(1)

    to_remove = torch.any(significantly_contained & is_smaller_than_others, dim=1)

    return torch.where(to_remove)[0].tolist()


def mask_drop_indices(masks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    indices_to_keep = torch.ones(masks.size(0), dtype=torch.bool, device=masks.device)
    indices_to_keep[indices] = 0
    return masks[indices_to_keep]


if __name__ == "__main__":

    def results_with_mask():
        orig_shape = (200, 200)

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

        masks = [mask_a, mask_c, mask_d]

        stacked_masks = torch.tensor(np.stack(masks), dtype=torch.bool)

        return stacked_masks

    result = results_with_mask()

    drop_id = torch_mask_nms(result)

    masks = mask_drop_indices(result, drop_id)

    print(masks)
