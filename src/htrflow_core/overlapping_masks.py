# Code that filters the segments based on threshold should be put here.
import torch

from htrflow_core.results import Result


def find_overlapping_masks_to_remove(result: Result, containments_threshold=0.5, batch_size=100):
    num_masks = result.masks.size(0)
    containments = torch.zeros((num_masks, num_masks), device=result.device)

    # Batch calculation of containments
    for i in range(0, num_masks, batch_size):
        batch_masks = result[i : i + batch_size]
        for j in range(num_masks):
            containments[i : i + batch_size, j] = calculate_containment_mask(batch_masks, result[j])

    # Determine masks to drop
    drop_indices = []
    for i in range(num_masks):
        # Masks containing the current mask above the threshold
        containing_masks = containments[:, i] > containments_threshold
        containing_masks[i] = False  # Exclude self-containment
        if containing_masks.any():
            drop_indices.append(i)

    return drop_indices


def calculate_containment_mask(masks_a, mask_b):
    intersections = torch.logical_and(masks_a, mask_b.unsqueeze(0)).sum(dim=(1, 2)).float()
    containments = intersections / mask_b.sum().float() if mask_b.sum() > 0 else torch.tensor(0.0)
    return containments


if __name__ == "__main__":
    import cv2

    from htrflow_core.models.openmmlab.rmtdet import RTMDet
    from htrflow_core.utils.image import helper_plot_for_segment

    model = RTMDet(
        model="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/model.pth",
        config="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/config.py",
        device="cuda:0",
    )

    img2 = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    image2 = cv2.imread(img2)

    results = model([image2], pred_score_thr=0.4)

    index_to_drop = find_overlapping_masks_to_remove(results)

    new_results = results.drop(index_to_drop)

    helper_plot_for_segment(image2, new_results[0].segments, maskalpha=0.7, boxcolor=None)

    # TODO test so this always return corrrect format to Results
    # TODO pytest
    # TODO fix openmmlabloader and hfdownloader
    # Fix overlpapping_mask
