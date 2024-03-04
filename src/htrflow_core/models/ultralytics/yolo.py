from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results

from htrflow_core.models.ultralytics.ultralytics_model import UltralyticsBaseModel
from htrflow_core.results import SegmentationResult


class YOLO(UltralyticsBaseModel):
    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        hf_token: Optional[str] = None,
        cache_dir: str | Path = "./.cache",
        *args,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        model = self._load_model(model)

        self.model = UltralyticsYOLO(model, *args)
        self.metadata = {self.META_MODEL_TYPE: str(model)}

    def _load_model(self, model):
        model_path = Path(model)
        try:
            if model_path.suffix not in self.SUPPORTED_MODEL_TYPES:
                model = self._download_from_hub(model)
            elif not model_path.exists():
                raise FileNotFoundError(f"Model file {model} not found.")
        except Exception as e:
            raise NotImplementedError(
                f"Unable to load model='{model}'. As an example, try load models with supported types:{self.SUPPORTED_MODEL_TYPES}."
            ) from e

        return model

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[SegmentationResult]:
        outputs: List[Results] = self.model(images, stream=True, **kwargs)

        return [self._create_segmentation_result(image, output) for image, output in zip(images, outputs)]

    def _create_segmentation_result(self, image: np.ndarray, output: Results) -> SegmentationResult:
        boxes = [[x1, x2, y1, y2] for x1, y1, x2, y2 in output.boxes.xyxy.int().tolist()]
        scores = output.boxes.conf.tolist()
        class_labels = [output.names[label] for label in output.boxes.cls.tolist()]
        return SegmentationResult.from_bboxes(self.metadata, image, boxes, scores, class_labels)


if __name__ == "__main__":
    import cv2

    model = YOLO("ultralyticsplus/yolov8s")  # ultralyticsplus/yolov8s

    img = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    image = cv2.imread(img)

    # print(model([image]))
