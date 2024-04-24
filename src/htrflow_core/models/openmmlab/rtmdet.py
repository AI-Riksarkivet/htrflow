import logging

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import MMLabsDownloader
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.models.torch_mixin import PytorchMixin
from htrflow_core.postprocess.mask_nms import multiclass_mask_nms
from htrflow_core.results import Result, Segment
from htrflow_core.utils.imgproc import resize


logger = logging.getLogger(__name__)


class RTMDet(BaseModel, PytorchMixin):
    def __init__(
        self,
        model: str = "Riksarkivet/rtmdet_regions",
        config: str = "Riksarkivet/rtmdet_regions",
        *model_args,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        model_weights, model_config = MMLabsDownloader.from_pretrained(model, config, self.cache_dir, self.hf_token)

        with SuppressOutput():
            self.model = DetInferencer(
                model=model_config,
                weights=model_weights,
                device=self.set_device(self.device),
                show_progress=False,
                *model_args,
            )

        logger.info(f"Model loaded ({self.device}) from {model}.")

        self.metadata.update(
            {
                "model": str(model),
                "config": str(config),
                "framework": Framework.Openmmlab.value,
                "task": [Task.ObjectDetection.value, Task.InstanceSegmentation.value],
                "device": self.device,
            }
        )

    def _predict(self, images: list[np.ndarray], nms_downscale=1, **kwargs) -> list[Result]:
        if len(images) > 1:
            batch_size = len(images)
        else:
            batch_size = 1

        outputs = self.model(images, batch_size=batch_size, draw_pred=False, return_datasample=True, **kwargs)
        results = []
        for image, output in zip(images, outputs["predictions"]):
            results.append(self._create_segmentation_result(image, output, nms_downscale))
        return results

    def _create_segmentation_result(self, image: np.ndarray, output: DetDataSample, nms_downscale: float) -> Result:
        sample: InstanceData = output.pred_instances
        boxes = sample.bboxes.int().tolist()

        masks = self._create_masks_and_test_alignment(image, sample)

        scores = sample.scores.tolist()
        class_labels = sample.labels.tolist()

        segments = [
            Segment(bbox=box, mask=mask, score=score, class_label=class_label)
            for box, mask, score, class_label in zip(boxes, masks, scores, class_labels)
        ]

        result = Result.segmentation_result(image, self.metadata, segments)
        indices_to_drop = multiclass_mask_nms(result, downscale=nms_downscale)
        result.drop_indices(indices_to_drop)

        logger.info("Found %d segments, dropped %d", len(scores), len(indices_to_drop))
        return result

    def _create_masks_and_test_alignment(self, image, sample):
        masks = self.to_numpy(sample.masks).astype(np.uint8)
        _, *mask_size = masks.shape  # n_masks, (height, width)
        *image_size, _ = image.shape  # (height, width), n_channels
        if mask_size != image_size:
            msg = "Mask and image shape not equal (masks %d-by-%d, image %d-by-%d). Resizing masks."
            logger.warning(msg, *mask_size, *image_size)
            masks = [resize(mask, image_size) for mask in masks]
        return masks


if __name__ == "__main__":
    model = RTMDet("Riksarkivet/rtmdet_lines")

    img = "/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/data/demo_images/demo_image.jpg"

    results = model([img] * 1, batch_size=1)

    print(results[0])
