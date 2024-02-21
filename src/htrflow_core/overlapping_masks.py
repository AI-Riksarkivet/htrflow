# Code that filters the segments based on threshold should be put here.
import torch
from torch import Tensor


class SegResult:
    def __init__(
        self, labels: Tensor, scores: Tensor, bboxes: Tensor = None, masks: Tensor = None, polygons: list = None
    ):
        self.labels = labels
        self.scores = scores
        self.bboxes = bboxes
        self.masks = masks
        self.polygons = polygons

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



# class PostProcessSegmentation:
#     def __init__(self):
#         pass

#     def get_bounding_box(mask):
#         rows = torch.any(mask, dim=1)
#         cols = torch.any(mask, dim=0)
#         ymin, ymax = torch.where(rows)[0][[0, -1]]
#         xmin, xmax = torch.where(cols)[0][[0, -1]]

#         return xmin, ymin, xmax, ymax

#     @staticmethod
#     @timing_decorator
#     def crop_imgs_from_result_optim(result: Result, img):
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         # Convert img to a PyTorch tensor and move to GPU if available
#         img = torch.from_numpy(img).to(device)

#         cropped_imgs = []
#         masks = result.segmentation.masks.to(device)

#         for mask in masks:
#             # Get bounding box
#             xmin, ymin, xmax, ymax = PostProcessSegmentation.get_bounding_box(mask)

#             # Crop masked region and put on white background
#             masked_region = img[ymin : ymax + 1, xmin : xmax + 1]
#             white_background = torch.ones_like(masked_region) * 255

#             # Apply mask to the image
#             masked_region_on_white = torch.where(
#                 mask[ymin : ymax + 1, xmin : xmax + 1][..., None], masked_region, white_background
#             )
#             masked_region_on_white_np = masked_region_on_white.cpu().numpy()

#             cropped_imgs.append(masked_region_on_white_np)

#         return cropped_imgs

#     def combine_region_line_res(result_full, result_regions):
#         ind = 0

#         for res in result_full:
#             res.nested_results = []
#             for i in range(ind, ind + len(res.segmentation.masks)):
#                 # result_lines.parent_result = res
#                 res.nested_results.append(result_regions[i])

#             ind += len(res.segmentation.masks)
