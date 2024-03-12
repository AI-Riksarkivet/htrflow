# https://mmdetection.readthedocs.io/en/main/user_guides/inference.html?highlight=Detinferencer#basic-usage
# TODO  opnemlabdownloader and hf_downloader needs to be changed.. Since what happend when we pass both config and model into here..,
# than we should not download both or either..
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample

from htrflow_core.models.base_model import BaseModel
from htrflow_core.results import Result


warnings.filterwarnings("ignore")


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

        self.model = DetInferencer(
            model=config, weights=model, device=self._device(device), show_progress=False, *args
        )
        self.metadata = {"model": str(model)}

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        if len(images) > 1:
            batch_size = len(images)
        else:
            batch_size = 1

        outputs = self.model(images, batch_size=batch_size, draw_pred=False, return_datasample=True, **kwargs)
        self._create_segmentation_result(images, outputs)

        return "hej"

    def _create_segmentation_result(self, images: list[np.ndarray], outputs: DetDataSample) -> Result:
        for output, image in zip(outputs["predictions"], images):
            labels = output.pred_instances.labels.clone()
            bboxes = output.pred_instances.bboxes.clone()
            masks = output.pred_instances.masks.clone()
            scores = output.pred_instances.scores.clone()
            print(labels, bboxes, masks, scores, image)

        # batch_result = [
        #             segments=Segment(
        #                 labels=x.pred_instances.labels.clone(),
        #                 bboxes=x.pred_instances.bboxes.clone(),
        #                 masks=x.pred_instances.masks.clone(),
        #                 scores=x.pred_instances.scores.clone(),
        #             ),
        #         )
        #         for x, y in zip(output["predictions"]
        #     ]


#     @timing_decorator
#     def postprocess(self, result_raw, input_images):
#         pass


#     #     batch_result = [
#     #         Result(
#     #             img_shape=np.shape(y)[0:2],
#     #             segmentation=SegResult(
#     #                 labels=x.pred_instances.labels.clone(),
#     #                 bboxes=x.pred_instances.bboxes.clone(),
#     #                 masks=x.pred_instances.masks.clone(),
#     #                 scores=x.pred_instances.scores.clone(),
#     #             ),
#     #         )
#     #         for x, y in zip(result_raw["predictions"], input_images)
#     #     ]

#     #     return batch_result
#     def align_masks_with_image(self, img):
#         # Create a tensor of all masks
#         all_masks = self.masks

#         # Calculate the size of the image
#         img_size = (img.shape[0], img.shape[1])

#         # Resize and pad each mask to match the size of the image
#         masks = []
#         for i in range(all_masks.shape[0]):
#             mask = all_masks[i]

#             # Convert the mask to float
#             mask_float = mask.float()

#             # Resize the mask
#             mask_resized = torch.nn.functional.interpolate(mask_float[None, None, ...], size=img_size,
# mode="nearest")[
#                 0, 0
#             ]

#             # Convert the mask back to bool
#             mask = mask_resized.bool()

#             # Pad the mask
#             padded_mask = torch.zeros(img_size, dtype=torch.bool, device=mask.device)
#             padded_mask[: mask.shape[0], : mask.shape[1]] = mask
#             mask = padded_mask

#             masks.append(mask)

#         # Stack all masks into a single tensor
#         self.masks = torch.stack(masks)


if __name__ == "__main__":
    import cv2

    model = RTMDet(
        model="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/model.pth",
        config="/home/gabriel/Desktop/htrflow_core/.cache/models--Riksarkivet--rtmdet_lines/snapshots/41a37f829aa3bb0d6997dbaa9eeacfe8bd767cfa/config.py",
        device="cuda:0",
    )

    img = "/home/gabriel/Desktop/htrflow_core/data/trocr_demo_image.png"
    img2 = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    image = cv2.imread(img)
    image2 = cv2.imread(img2)

    results = model([image, image2], batch_size=1)
