from os import PathLike
from typing import Optional

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.openmmlab import openmmlab_downloader
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.models.torch_mixin import PytorchDeviceMixin
from htrflow_core.results import Result, Segment


class RTMDet(BaseModel, PytorchDeviceMixin):
    def __init__(
        self,
        model: str | PathLike = "Riksarkivet/rtmdet_regions",
        config: str | PathLike = "Riksarkivet/rtmdet_regions",
        device: Optional[str] = None,
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
        *args,
    ) -> None:
        self.cache_dir = cache_dir

        model_weights, model_config = openmmlab_downloader.load_from_hf(model, config, cache_dir, hf_token)

        with SuppressOutput():
            self.model = DetInferencer(
                model=model_config, weights=model_weights, device=self.set_device(device), show_progress=False, *args
            )

        self.metadata = {"model": str(model), "config": str(config)}

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
        boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in sample.bboxes.int().tolist()]
        masks = self.to_numpy(sample.masks).astype(np.uint8)
        scores = sample.scores.tolist()
        class_labels = sample.labels.tolist()

        segments = [
            Segment(bbox=box, mask=mask, score=score, class_label=class_label)
            for box, mask, score, class_label in zip(boxes, masks, scores, class_labels)
        ]

        return Result.segmentation_result(image, self.metadata, segments)


if __name__ == "__main__":
    img = "/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/data/demo_images/trocr_demo_image.png"

    model = RTMDet(
        model="Riksarkivet/rtmdet_regions",
    )

    results = model([img] * 2, batch_size=2)
