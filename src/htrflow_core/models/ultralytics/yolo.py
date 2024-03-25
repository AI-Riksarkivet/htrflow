from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results as UltralyticsResults

from htrflow_core.models.base_model import BaseModel
from htrflow_core.models.ultralytics.ultralytics_downloader import UltralyticsDownloader
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
        super().__init__(device=device)
        self.cache_dir = cache_dir
        model_file = UltralyticsDownloader.from_pretrained(model, cache_dir, hf_token)
        self.model = UltralyticsYOLO(model_file, *args).to(self.device)
        self.metadata = {"model": str(model)}

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[Result]:
        outputs = self.model(images, stream=True, verbose=False, **kwargs)
        return [self._create_segmentation_result(image, output) for image, output in zip(images, outputs)]

    def _create_segmentation_result(self, image: np.ndarray, output: UltralyticsResults) -> Result:
        if output.boxes is not None:
            boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in output.boxes.xyxy.int().tolist()]
            scores = output.boxes.conf.tolist()
            class_labels = [output.names[label] for label in output.boxes.cls.tolist()]
        if output.masks is not None:
            masks = self._create_masks_from_polygons(image, output.masks.xy)
        else:
            masks = [None] * len(boxes)

        segments = [
            Segment(bbox=box, mask=mask, score=score, class_label=class_label)
            for box, mask, score, class_label in zip(boxes, masks, scores, class_labels)
        ]

        return Result.segmentation_result(image, self.metadata, segments)

    def _create_masks_from_polygons(self, image, polygons_xy):
        image_height, image_width = image.shape[:2]
        masks = []

        for xy in polygons_xy:
            polygon = np.round(xy).astype(np.int32)
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], color=255)
            masks.append(mask)

        return masks
