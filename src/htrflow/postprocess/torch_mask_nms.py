from typing import List

import torch


# TODO: torch_mask_nms
def multiclass_mask_nms():
    pass


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
    is_smaller_than_others = mask_areas.unsqueeze(0) < mask_areas.unsqueeze(1)

    to_remove = torch.any(significantly_contained & is_smaller_than_others, dim=1)

    return torch.where(to_remove)[0].tolist()


def mask_drop_indices(masks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    indices_to_keep = torch.ones(masks.size(0), dtype=torch.bool, device=masks.device)
    indices_to_keep[indices] = 0
    return masks[indices_to_keep]
