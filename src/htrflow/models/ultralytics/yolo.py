import logging

import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results as UltralyticsResults

from htrflow.models.base_model import BaseModel
from htrflow.models.hf_utils import commit_hash_from_path, load_ultralytics
from htrflow.results import Result


logger = logging.getLogger(__name__)


class YOLO(BaseModel):
    def __init__(self, model: str, revision: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)

        model_file = load_ultralytics(model, revision)
        self.model = UltralyticsYOLO(model_file).to(self.device)

        logger.info(
            "Initialized YOLO model '%s' from %s on device %s",
            model,
            model_file,
            self.model.device,
        )

        self.metadata.update({"model": model, "model_version": commit_hash_from_path(model_file)})

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
