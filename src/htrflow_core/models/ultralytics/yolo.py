import logging
from os import PathLike

import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results as UltralyticsResults

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import UltralyticsDownloader
from htrflow_core.models.torch_mixin import PytorchMixin
from htrflow_core.results import Result, Segment
from htrflow_core.utils.geometry import polygons2masks


logger = logging.getLogger(__name__)


class YOLO(BaseModel, PytorchMixin):
    def __init__(self, model: str | PathLike = "ultralyticsplus/yolov8s", *model_args, **kwargs) -> None:
        super().__init__(**kwargs)

        model_file = UltralyticsDownloader.from_pretrained(model, self.cache_dir)
        self.model = UltralyticsYOLO(model_file, *model_args).to(self.set_device(self.device))

        logger.info(f"Model loaded ({self.device}) from {model}.")

        self.metadata.update(
            {
                "model": str(model),
                "framework": Framework.Ultralytics.value,
                "task": [Task.ObjectDetection.value, Task.InstanceSegmentation.value],
                "device": self.device,
            }
        )

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        outputs = self.model(images, stream=True, verbose=False, **kwargs)
        return [self._create_segmentation_result(image, output) for image, output in zip(images, outputs)]

    def _create_segmentation_result(self, image: np.ndarray, output: UltralyticsResults) -> Result:
        if output.boxes is not None:
            boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in output.boxes.xyxy.int().tolist()]
            scores = output.boxes.conf.tolist()
            class_labels = [output.names[label] for label in output.boxes.cls.tolist()]

        if output.masks is not None:
            masks = polygons2masks(image, output.masks.xy)
        else:
            masks = [None] * len(boxes)

        segments = [
            Segment(bbox=box, mask=mask, score=score, class_label=class_label)
            for box, mask, score, class_label in zip(boxes, masks, scores, class_labels)
        ]

        return Result.segmentation_result(image, self.metadata, segments)
