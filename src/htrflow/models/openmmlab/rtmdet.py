import logging

import torch
import torch.nn.functional as F
from mmdet.apis import DetInferencer
from mmdet.models.layers.matrix_nms import mask_matrix_nms
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow.models.base_model import BaseModel
from htrflow.models.hf_utils import commit_hash_from_path, load_mmlabs
from htrflow.models.openmmlab.utils import SuppressOutput
from htrflow.results import Result
from htrflow.utils.imgproc import NumpyImage, resize


logger = logging.getLogger(__name__)


class RTMDet(BaseModel):
    """
    HTRFLOW adapter of Openmmlabs' RTMDet model

    This model can be used for region and line segmentation. Riksarkivet
    provides two pre-trained RTMDet models:

        -   https://huggingface.co/Riksarkivet/rtmdet_lines
        -   https://huggingface.co/Riksarkivet/rtmdet_regions
    """

    def __init__(
        self,
        model: str,
        config: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> None:
        """
        Arguments:
            model: Path to a local .pth model weights file or to a
                huggingface repo which contains a .pth file, for example
                'Riksarkivet/rtmdet_lines'.
            config: Path to a local config.py file or to a huggingface
                repo which contains a config.py file, for example
                'Riksarkivet/rtmdet_lines'.
            revision: A specific model revision, as a commit hash of the
                model's huggingface repo. If None, the latest available
                revision is used.
            kwargs: Additional kwargs which are forwarded to BaseModel's
                __init__.
        """
        super().__init__(**kwargs)

        config = config or model
        model_weights, model_config = load_mmlabs(model, config, revision)

        with SuppressOutput():
            self.model = DetInferencer(
                model=model_config,
                weights=model_weights,
                device=self.device,
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
                "model": model,
                "model_version": commit_hash_from_path(model_weights),
                "config": config,
                "config_version": commit_hash_from_path(model_config),
            }
        )

    def _predict(
        self,
        images: list[NumpyImage],
        nms_downscale: float = 1.0,
        nms_threshold: float = 0.4,
        nms_sigma: float = 2.0,
        **kwargs,
    ) -> list[Result]:
        """
        RTMDet-specific prediction method

        This method is used by `predict()` and should typically not be
        called directly.

        Arguments:
            images: List of input images
            nms_downscale: If < 1, all masks will be downscaled by this factor
                before applying NMS. This leads to faster NMS at the expense of
                accuracy.
            nms_threshold: Score threshold for segments to keep after NMS.
            nms_sigma: NMS parameter that affects the score calculation.
            **kwargs: Additional arguments that are passed to DetInferencer.__call__.
        """
        batch_size = max(1, len(images))
        outputs = self.model(
            images,
            batch_size=batch_size,
            draw_pred=False,
            return_datasample=True,
            **kwargs,
        )
        results = []
        for image, output in zip(images, outputs["predictions"]):
            results.append(self._create_segmentation_result(image, output, nms_downscale, nms_threshold, nms_sigma))
        return results

    def _create_segmentation_result(
        self,
        image: NumpyImage,
        output: DetDataSample,
        nms_downscale: float,
        nms_threshold: float,
        nms_sigma: float,
    ) -> Result:
        sample: InstanceData = output.pred_instances

        # Cast masks to uint8 (needed for F.interpolate and Result obj)
        masks = sample.masks.to(torch.uint8)

        # Apply mask NMS
        downscaled_masks = F.interpolate(masks, scale_factor=nms_downscale).to(torch.bool)
        scores, labels, _, keep_inds = mask_matrix_nms(
            downscaled_masks,
            sample.labels,
            sample.scores,
            sigma=nms_sigma,
            filter_thr=nms_threshold,
        )

        # RTMDet sometimes return masks of slightly different shape (+- a few pixels)
        # than the input image. To avoid alignment problems later on, all masks are
        # resized to the original image shape.
        orig_shape = image.shape[:2]
        masks = [resize(mask, orig_shape) for mask in masks[keep_inds].cpu().numpy()]

        result = Result.segmentation_result(
            orig_shape,
            bboxes=sample.bboxes[keep_inds].int().tolist(),
            masks=masks,
            scores=scores.tolist(),
            labels=labels.tolist(),
            metadata=self.metadata,
        )

        logger.info(
            "%s (%d x %d): Found %d segments, kept %d after NMS",
            getattr(image, "name", "unlabelled image"),
            *orig_shape,
            len(sample.scores),
            len(result.segments),
        )
        return result
