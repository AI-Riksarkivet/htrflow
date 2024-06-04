import logging

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import MMLabsDownloader
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.postprocess.mask_nms import multiclass_mask_nms
from htrflow_core.results import Result
from htrflow_core.utils.imgproc import Mask, NumpyImage, resize


logger = logging.getLogger(__name__)


class RTMDet(BaseModel, PytorchMixin):
    def __init__(self, model: str, config: str | None = None, device: str | None = None) -> None:
        super().__init__(device)

        model_weights, model_config = MMLabsDownloader.from_pretrained(model, config)

        with SuppressOutput():
            self.model = DetInferencer(
                model=model_config,
                weights=model_weights,
                device=self.set_device(self.device),
                show_progress=False,
            )

        logger.info(
            "Loaded RTMDet model '%s' from %s with config %s on device %s",
            model,
            model_weights,
            model_config,
            self.device,
        )

        self.metadata.update(
            {
                "model": str(model),
                "config": str(config),
                "framework": Framework.Openmmlab.value,
                "task": [Task.ObjectDetection.value, Task.InstanceSegmentation.value],
            }
        )

    def _predict(self, images: list[NumpyImage], nms_downscale: float = 1.0, **kwargs) -> list[Result]:
        batch_size = max(1, len(images))
        outputs = self.model(images, batch_size=batch_size, draw_pred=False, return_datasample=True, **kwargs)
        results = []
        for image, output in zip(images, outputs["predictions"]):
            results.append(self._create_segmentation_result(image, output, nms_downscale))
        return results

    def _create_segmentation_result(self, image: NumpyImage, output: DetDataSample, nms_downscale: float) -> Result:
        sample: InstanceData = output.pred_instances
        boxes = sample.bboxes.int().tolist()

        masks = self._create_masks_and_test_alignment(image, sample)

        scores = sample.scores.tolist()
        class_labels = sample.labels.tolist()

        result = Result.segmentation_result(
            image.shape[:2], bboxes=boxes, masks=masks, scores=scores, labels=class_labels, metadata=self.metadata
        )
        indices_to_drop = multiclass_mask_nms(result, downscale=nms_downscale)
        result.drop_indices(indices_to_drop)

        logger.info(
            "%s (%d x %d): Found %d segments, dropped %d",
            getattr(image, "name", "unlabelled image"),
            *image.shape[:2],
            len(scores),
            len(indices_to_drop),
        )
        return result

    def _create_masks_and_test_alignment(self, image: NumpyImage, sample: InstanceData) -> list[Mask]:
        masks = self.to_numpy(sample.masks).astype(np.uint8)
        _, *mask_size = masks.shape  # n_masks, (height, width)
        *image_size, _ = image.shape  # (height, width), n_channels
        if mask_size != image_size:
            logger.warning(
                "Mask and image shape not equal (masks %d-by-%d, image %d-by-%d). Resizing masks.",
                *mask_size,
                *image_size,
            )
            masks = [resize(mask, image_size) for mask in masks]
        return masks
