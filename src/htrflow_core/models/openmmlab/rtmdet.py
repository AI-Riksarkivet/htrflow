from os import PathLike

import numpy as np
from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from htrflow_core.models import hf_utils
from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.openmmlab.utils import SuppressOutput
from htrflow_core.models.torch_mixin import PytorchMixin
from htrflow_core.postprocess.mask_nms import multiclass_mask_nms
from htrflow_core.results import Result, Segment


class RTMDet(BaseModel, PytorchMixin):
    def __init__(
        self,
        model: str | PathLike = "Riksarkivet/rtmdet_regions",
        config: str | PathLike = "Riksarkivet/rtmdet_regions",
        *model_args,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        model_weights, model_config = hf_utils.mmlabs_from_hf(model, config, self.cache_dir, self.hf_token)

        with SuppressOutput():
            self.model = DetInferencer(
                model=model_config,
                weights=model_weights,
                device=self.set_device(self.device),
                show_progress=False,
                *model_args,
            )

        self.metadata.update(
            {
                "model": str(model),
                "config": str(config),
                "framework": Framework.Openmmlab.value,
                "task": [Task.ObjectDetection.value, Task.InstanceSegmentation.value],
                "device": self.device,
            }
        )

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
        boxes = sample.bboxes.int().tolist()
        masks = self.to_numpy(sample.masks).astype(np.uint8)
        scores = sample.scores.tolist()
        class_labels = sample.labels.tolist()

        segments = [
            Segment(bbox=box, mask=mask, score=score, class_label=class_label)
            for box, mask, score, class_label in zip(boxes, masks, scores, class_labels)
        ]

        result = Result.segmentation_result(image, self.metadata, segments)
        indices_to_drop = multiclass_mask_nms(result)
        result.drop_indices(indices_to_drop)

        return result


if __name__ == "__main__":
    from htrflow_core.utils.draw import helper_plot_for_segment

    model = RTMDet("Riksarkivet/rtmdet_lines")

    img = "/home/adm.margabo@RA-ACC.INT/repo/htrflow_core/data/demo_images/demo_image.jpg"

    results = model([img] * 100, batch_size=16)

    print(results[0])

    helper_plot_for_segment(results[0].segments, results[0].image)
