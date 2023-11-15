import torch
from torch import Tensor

from htrflow.helper.timing_decorator import timing_decorator

# import cv2
# import numpy as np


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
    def align_masks_with_image(self, img):
        # Create a tensor of all masks
        all_masks = self.masks

        # Calculate the size of the image
        img_size = (img.shape[0], img.shape[1])

        # Resize and pad each mask to match the size of the image
        masks = []
        for i in range(all_masks.shape[0]):
            mask = all_masks[i]

            # Convert the mask to float
            mask_float = mask.float()

            # Resize the mask
            mask_resized = torch.nn.functional.interpolate(mask_float[None, None, ...], size=img_size, mode="nearest")[
                0, 0
            ]

            # Convert the mask back to bool
            mask = mask_resized.bool()

            # Pad the mask
            padded_mask = torch.zeros(img_size, dtype=torch.bool, device=mask.device)
            padded_mask[: mask.shape[0], : mask.shape[1]] = mask
            mask = padded_mask

            masks.append(mask)

        # Stack all masks into a single tensor
        self.masks = torch.stack(masks)
