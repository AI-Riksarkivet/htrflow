import logging

import cv2
import numpy as np
from ultralytics import YOLO as UltralyticsYOLO

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

        results = []
        for output in outputs:
            if output.boxes is None:
                results.append([])

            shapes = map(Bbox, output.boxes.xyxy.int().tolist())
            scores = output.boxes.conf.tolist()
            labels = [output.names[label] for label in output.boxes.cls.tolist()]

            if output.masks is not None:
                shapes = map(Polygon, _simplify_polygons(output.masks.xy, polygon_approx_level))

            result = [
                Region(shape, segmentation_confidence=score, segmentation_label=label)
                for shape, score, label in zip(shapes, scores, labels)
            ]
            results.append(result)
        return results


def _simplify_polygons(polygons, approx_level):
    result = []

    for polygon in polygons:
        # Ensure polygons are at least four points by replacing bad
        # polygons with None to use the bounding box instead.
        if len(polygon) < 4:
            result.append(None)
            continue

        perimeter = cv2.arcLength(polygon, True)
        approx = cv2.approxPolyDP(polygon, approx_level * perimeter, True)
        if len(approx) < 4:
            logger.warning(
                "A %d-point polygon was approximated to %d points with `approx_level`=%f. Consider using a lower"
                " `approx_level`.",
                len(polygon),
                len(approx),
                approx_level,
            )

        result.append(approx.squeeze().astype(np.int32))
    return result
