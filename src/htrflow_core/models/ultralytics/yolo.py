from pathlib import Path
from typing import List, Optional

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO as UltraylticsYOLO
from ultralytics.engine.results import Results

from htrflow_core.models.base_model import BaseModel
from htrflow_core.results import SegmentationResult


class YOLO(BaseModel):
    META_MODEL_TYPE = "model"
    CONFIG_FILE = "config.json"
    SUPPORTED_MODEL_TYPES = (".pt", ".yaml")

    def __init__(
        self,
        model: str | Path = "yolov8n.pt",
        hf_token: Optional[str] = None,
        cache_dir: str | Path = "./.cache",
        *args,
    ) -> None:
        model_path = Path(model)
        self.cache_dir = Path(cache_dir)

        try:
            if model_path.suffix not in YOLO.SUPPORTED_MODEL_TYPES:
                model = self.download_from_hub(model, hf_token=hf_token)
            elif not model_path.exists():
                raise FileNotFoundError(f"Model file {model} not found.")
        except Exception as e:
            raise NotImplementedError(
                f"Unable to load model='{model}'. As an example, try load models with supported types:{YOLO.SUPPORTED_MODEL_TYPES}."
            ) from e

        self.model = UltraylticsYOLO(model, *args)
        self.metadata = {YOLO.META_MODEL_TYPE: str(model)}

    def _predict(self, images: list[np.ndarray], **kwargs) -> list[SegmentationResult]:
        outputs: List[Results] = self.model(images, stream=True, **kwargs)

        return [self._create_segmentation_result(image, output) for image, output in zip(images, outputs)]

    def _create_segmentation_result(self, image: np.ndarray, output: Results) -> SegmentationResult:
        boxes = [[x1, x2, y1, y2] for x1, y1, x2, y2 in output.boxes.xyxy.int().tolist()]
        scores = output.boxes.conf.tolist()
        class_labels = [output.names[label] for label in output.boxes.cls.tolist()]
        return SegmentationResult.from_bboxes(self.metadata, image, boxes, scores, class_labels)

    def download_from_hub(self, hf_model_id: str, hf_token: str) -> Path:
        repo_files = list_repo_files(repo_id=hf_model_id, repo_type=YOLO.META_MODEL_TYPE, token=hf_token)

        if YOLO.CONFIG_FILE in repo_files:
            _ = self._hf_hub_download( repo_id = hf_model_id, filename= YOLO.CONFIG_FILE, YOLO.META_MODEL_TYPE, hf_token)

        model_file = next((f for f in repo_files if any(f.endswith(ext) for ext in YOLO.SUPPORTED_MODEL_TYPES)), None)
        if not model_file:
            raise ValueError(
                f"No model file of supported type: {YOLO.SUPPORTED_MODEL_TYPES} found in repository {hf_model_id}."
            )

        return self._hf_hub_download(hf_model_id, model_file, hf_token)

    def _hf_hub_download(self, hf_model_id, filename, repo_type, cache_dir ,hf_token):
        return hf_hub_download(
            repo_id=hf_model_id,
            filename=filename,
            repo_type=repo_type
            cache_dir=self.cache_dir,
            token=hf_token,
        )

    # TODO: Create a specif download download for openmmlab and ultralitics that extend basemodel clss
    # TODO: Test that models aand path for models works in try except..


if __name__ == "__main__":
    import cv2

    model = YOLO("ultralyticsplus/yolov8s.yaml")

    img = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    image = cv2.imread(img)

    print(model([image]))
