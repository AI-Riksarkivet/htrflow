from pathlib import Path
from typing import Optional

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.openmmlab.openmmlab_utils import SuppressOutput
from htrflow_core.results import Result, Segment


class RTMDet(BaseModel):
    def __init__(
        self,
        model: str | Path = "model.pth",
        config: str | Path = "config.py",
        device: str = "cuda",
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
        *args,
    ) -> None:
        super().__init__(device=device)

        self.cache_dir = cache_dir
        # config_py, weights = OpenmmlabDownloader.from_pretrained(model, cache_dir, hf_token)
        with SuppressOutput():
            self.model = DetInferencer(model=config, weights=model, device=self.device, show_progress=False, *args)

        self.metadata = {"model": str(model)}

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        if len(images) > 1:
            batch_size = len(images)
        else:
            batch_size = 1

        outputs = self.model(images, batch_size=batch_size, draw_pred=False, return_datasample=True, **kwargs)

        return [
            self._create_segmentation_result(image, output) for image, output in zip(images, outputs["predictions"])
        ]

    def _create_segmentation_result(self, image: np.ndarray, output: DetDataSample) -> Result:
        sample: InstanceData = output.pred_instances
        boxes = [[x1, x2, y1, y2] for x1, y1, x2, y2 in sample.bboxes.int().tolist()]
        masks = self.to_numpy(sample.masks).astype(np.uint8)
        scores = sample.scores.tolist()
        class_labels = sample.labels.tolist()

        segments = [
            Segment(bbox=box, mask=mask, score=score, class_label=class_label)
            for box, mask, score, class_label in zip(boxes, masks, scores, class_labels)
        ]

        return Result.segmentation_result(image, self.metadata, segments)

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


if __name__ == "__main__":
    import cv2

    from htrflow_core.utils.image import helper_plot_for_segment

    model = RTMDet(
        model="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/model.pth",
        config="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/config.py",
        device="cuda:0",
    )

    img2 = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    image2 = cv2.imread(img2)

    results = model([image2] * 1, pred_score_thr=0.4)

    helper_plot_for_segment(image2, results[0].segments, maskalpha=0.7, boxcolor=None)

    # TODO test so this always return corrrect format to Results
    # TODO pytest
    # TODO fix openmmlabloader and hfdownloader
    # Fix overlpapping_mask
