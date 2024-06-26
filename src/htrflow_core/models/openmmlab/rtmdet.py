import logging

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import load_mmlabs
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.postprocess.mask_nms import multiclass_mask_nms
from htrflow_core.results import Result
from htrflow_core.utils.imgproc import NumpyImage, resize


logger = logging.getLogger(__name__)


class RTMDet(BaseModel, PytorchMixin):
    """
    HTRFLOW adapter of Openmmlabs' RTMDet model

    This model can be used for region and line segmentation. Riksarkivet
    provides two pre-trained RTMDet models:
        -   https://huggingface.co/Riksarkivet/rtmdet_lines
        -   https://huggingface.co/Riksarkivet/rtmdet_regions
    """

    def __init__(self, model: str, config: str | None = None, device: str | None = None) -> None:
        """
        Initialize an RTMDet model.

        Arguments:
            model: Path to a local .pth model weights file or to a
                huggingface repo which contains a .pth file, for example
                'Riksarkivet/rtmdet_lines'.
            config: Path to a local config.py file or to a huggingface
                repo which contains a config.py file, for example
                'Riksarkivet/rtmdet_lines'.
            device: Model device.
        """
        super().__init__(device)

        model_weights, model_config = load_mmlabs(model, config)

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
        orig_shape = image.shape[:2]

        # RTMDet sometimes return masks of slightly different shape (+- a few pixels)
        # than the input image. To avoid alignment problems later on, all masks are
        # resized to the original image shape.
        np_masks = sample.masks.cpu().numpy().astype(np.uint8)
        masks = [resize(mask, orig_shape) for mask in np_masks]

        boxes = sample.bboxes.int().tolist()
        scores = sample.scores.tolist()
        class_labels = sample.labels.tolist()

        result = Result.segmentation_result(
            orig_shape, bboxes=boxes, masks=masks, scores=scores, labels=class_labels, metadata=self.metadata
        )
        indices_to_drop = multiclass_mask_nms(result, downscale=nms_downscale)
        result.drop_indices(indices_to_drop)

        logger.info(
            "%s (%d x %d): Found %d segments, dropped %d",
            getattr(image, "name", "unlabelled image"),
            *orig_shape,
            len(scores),
            len(indices_to_drop),
        )
        return result
