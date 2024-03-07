from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results as UltralyticsResults

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.ultralytics.ultralytics_model import UltralyticsModel
from htrflow_core.results import Result, Segment


class YOLO(BaseModel):
    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        device: str = "cuda",
        cache_dir: str = "./.cache",
        hf_token: Optional[str] = None,
        *args,
    ) -> None:
        self.cache_dir = cache_dir
        model_file = UltralyticsModel.from_pretrained(model, cache_dir, hf_token)
        self.model = UltralyticsYOLO(model_file, *args).to(self._device(device))
        self.metadata = {"model": str(model)}

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        outputs = self.model(images, stream=True, **kwargs)

        return [self._create_segmentation_result(image, output) for image, output in zip(images, outputs)]

    def _create_segmentation_result(self, image: np.ndarray, output: UltralyticsResults) -> Result:
        boxes = [[x1, x2, y1, y2] for x1, y1, x2, y2 in output.boxes.xyxy.int().tolist()]
        scores = output.boxes.conf.tolist()
        class_labels = [output.names[label] for label in output.boxes.cls.tolist()]

        segments = [
            Segment.from_bbox(box, score=score, class_label=class_label)
            for box, score, class_label in zip(boxes, scores, class_labels)
        ]

        return Result.segmentation_result(self.metadata, image, segments)


if __name__ == "__main__":
    import cv2

    model = YOLO(model="ultralyticsplus/yolov8s")

    img = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    image = cv2.imread(img)

    results = model([image] * 100)

    print(results)
