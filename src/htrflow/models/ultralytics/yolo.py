import logging

import cv2
import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.engine.results import Results

from htrflow.document import Region
from htrflow.models.base_model import BaseModel
from htrflow.models.download import commit_hash_from_path, load_ultralytics
from htrflow.utils.geometry import Bbox, Polygon


logger = logging.getLogger(__name__)


class YOLO(BaseModel):
    """
    HTRflow adapter of Ultralytics' YOLO model

    Example usage with the `Segmentation` step:
    ```yaml
    - step: Segmentation
      settings:
        model: YOLO
        model_settings:
          model: Riksarkivet/yolov9-regions-1
          revision: 7c44178d85926b4a096c55c89bf224855a201fbf
          device: cpu
        generation_settings:
          batch_size: 8
    ```

    `generation_settings` accepts the same arguments as `YOLO.predict()`.
    See the [Ultralytics documentation](https://docs.ultralytics.com/modes/predict/#inference-arguments)
    for a list of supported arguments.
    """

    def __init__(self, model: str, revision: str | None = None, **kwargs) -> None:
        """
        Arguments:
            model: Path to a YOLO model. The path can be a path to a
                local .pt model file (for example, `my-model.py`) or an
                indentifier of a Huggingface repo contatining a .pt
                model file (for example, `Riksarkivet/yolov9-regions-1`).
            revision: Optional revision of the Huggingface repository.
        """
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

    def _predict(self, images: list[np.ndarray], polygon_approx_level: float = 0.005, **kwargs) -> list[list[Region]]:
        """
        Run inference.

        Arguments:
            images: Input images
            polygon_approx_level: A parameter which controls the maximum distance between the original polygon
                and the approximated low-resolution polygon, as a fraction of the original polygon arc length.
                Example: With `polygon_approx_level=0.005` and a generated polygon with arc length 100, the
                approximated polygon will not differ more than 0.5 units from the original.
            **kwargs: Keyword arguments forwarded to the inner YOLO model instance.
        """
        outputs = self.model(images, stream=True, verbose=False, **kwargs)
        return [_postprocess(output, polygon_approx_level) for output in outputs]


def _postprocess(output: Results, polygon_approx_level: float) -> list[Region]:
    """
    Convert YOLO output to a list of `Region`
    """

    if output.boxes is None:
        return []

    # Get polygons of detections: Use the mask's polygon if available, fall
    # back to the bounding box elsewise.
    boxes = [Bbox(*bbox).polygon() for bbox in output.boxes.xyxy.int().tolist()]
    if output.masks is None:
        polygons = boxes
    else:
        polygons = []
        for bbox, polygon in zip(boxes, output.masks.xy):
            polygon = _simplify_polygon(polygon, polygon_approx_level)
            if len(polygon) > 3:
                polygon = Polygon(polygon)
            else:
                polygon = bbox
            polygons.append(polygon)

    scores = output.boxes.conf.tolist()
    labels = [output.names.get(label, None) for label in output.boxes.cls.tolist()]

    return [
        Region(polygon, segmentation_confidence=score, segmentation_label=label)
        for polygon, score, label in zip(polygons, scores, labels)
    ]


def _simplify_polygon(polygon, approx_level):
    perimeter = cv2.arcLength(polygon, True)
    approx = cv2.approxPolyDP(polygon, approx_level * perimeter, True)
    if approx is None:
        return []

    if len(approx) < 4:
        logger.warning(
            "A %d-point polygon was approximated to %d points with"
            "`approx_level`=%f. Consider using a lower `approx_level`.",
            len(polygon),
            len(approx),
            approx_level,
        )
    return approx.squeeze().astype(np.int32)
