import logging
from os import PathLike

import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results as UltralyticsResults

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.enums import Framework, Task
from htrflow_core.models.hf_utils import UltralyticsDownloader
from htrflow_core.models.mixins.torch_mixin import PytorchMixin
from htrflow_core.results import Result


logger = logging.getLogger(__name__)


class YOLO(BaseModel, PytorchMixin):
    def __init__(self, model: str | PathLike = "ultralyticsplus/yolov8s", *model_args, **kwargs) -> None:
        super().__init__(**kwargs)

        model_file = UltralyticsDownloader.from_pretrained(model)
        self.model = UltralyticsYOLO(model_file, *model_args).to(self.set_device(self.device))

        logger.info("Initialized YOLO model from %s on device %s", model, self.model.device)

        self.metadata.update(
            {
                "model": str(model),
                "framework": Framework.Ultralytics.value,
                "task": [Task.ObjectDetection.value, Task.InstanceSegmentation.value],
                "device": self.device_id,
            }
        )

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        outputs = self.model(images, stream=True, verbose=False, **kwargs)
        return [self._create_segmentation_result(image, output) for image, output in zip(images, outputs)]

    def _create_segmentation_result(self, image: np.ndarray, output: UltralyticsResults) -> Result:
        bboxes = scores = class_labels = None
        if output.boxes is not None:
            bboxes = output.boxes.xyxy.int().tolist()
            scores = output.boxes.conf.tolist()
            class_labels = [output.names[label] for label in output.boxes.cls.tolist()]

        polygons = output.masks.xy if output.masks is not None else []
        return Result.segmentation_result(
            image.shape[:2],
            bboxes=bboxes,
            polygons=polygons,
            scores=scores,
            labels=class_labels,
            metadata=self.metadata,
        )
