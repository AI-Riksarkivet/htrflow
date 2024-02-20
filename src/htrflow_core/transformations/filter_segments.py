# Code that filters the segments based on threshold should be put here.
import torch
from torch import Tensor

from htrflow_core.helper.timing_decorator import timing_decorator


# import cv2
# import numpy as np

# TODO: Perhaps should be move into here: https://github.com/viklofg/legendary-space-giggle/blob/1b063e47da10a6872b60d08a0c3497a70625bf60/results.py#L52


class SegResult:
    def __init__(
        self, labels: Tensor, scores: Tensor, bboxes: Tensor = None, masks: Tensor = None, polygons: list = None
    ):
        self.labels = labels
        self.scores = scores
        self.bboxes = bboxes
        self.masks = masks
        self.polygons = polygons

    @timing_decorator
    def remove_overlapping_masks(self, method="mask", containments_threshold=0.5):
        # Compute pairwise containment
        containments = torch.zeros((len(self.masks), len(self.masks)))

        # Create a tensor of all masks
        all_masks = self.masks

        # Calculate the area of each mask
        torch.sum(all_masks.view(len(self.masks), -1), dim=1)

        # Calculate containments in batches
        batch_size = 100  # adjust this value based on your GPU memory
        for i in range(0, len(self.masks), batch_size):
            batch_masks = all_masks[i : i + batch_size]
            for j in range(len(self.masks)):
                if method == "mask":
                    containments[i : i + batch_size, j] = self._calculate_containment_mask(batch_masks, self.masks[j])

        # Keep only the biggest masks for overlapping pairs
        keep_mask = torch.ones(len(self.masks), dtype=torch.bool)
        for i in range(len(self.masks)):
            if not keep_mask[i]:
                continue
            # Get indices of masks that contain mask i
            containing_indices = torch.where(
                (containments[:, i] > containments_threshold) & (torch.arange(len(self.masks)) != i)
            )[0]
            # Mark mask i for removal if it's contained in any other mask
            if len(containing_indices) > 0:
                keep_mask[i] = False

        self.masks = self.masks[keep_mask]
        self.labels = self.labels[keep_mask]
        self.bboxes = self.bboxes[keep_mask]
        self.scores = self.scores[keep_mask]

    def _calculate_containment_mask(self, masks_a, mask_b):
        intersections = torch.logical_and(masks_a, mask_b).sum(dim=(1, 2)).float()
        containments = intersections / mask_b.sum().float() if mask_b.sum() > 0 else 0
        return containments

    @timing_decorator
