import numpy as np

from htrflow_core.models.base_inferencer import BaseModel
from htrflow_core.results import SegmentationResult
from ultralyticsplus import YOLO as UltraylticsplusYOLO


class YOLO(BaseModel):
    def __init__(self, model, *args):
        self.model = UltraylticsplusYOLO(model, *args)
        self.metadata = {'model': model}

    def predict(self, images: list[np.ndarray], **kwargs) -> list[SegmentationResult]:
        outputs = self.model(images, stream=True, **kwargs)

        results = []
        for image, output in zip(images, outputs):
            boxes = [[x1, x2, y1, y2] for x1, y1, x2, y2 in output.boxes.xyxy.int()]
            scores = output.boxes.conf.tolist()
            class_labels = [output.names[label] for label in output.boxes.cls.tolist()]
            result = SegmentationResult.from_bboxes(self.metadata, image, boxes, scores, class_labels)
            results.append(result)
        return results
