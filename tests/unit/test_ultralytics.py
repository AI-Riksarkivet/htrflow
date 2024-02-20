import warnings

import pytest
from htrflow_core.inferencers.ultralytics.ultralytics_inferencer import UltralyticsInferencer
from ultralyticsplus import YOLO


@pytest.fixture
def inferencer():
    model = YOLO("ultralyticsplus/yolov8s")
    return UltralyticsInferencer(model)


@pytest.fixture
def image():
    # Ultralytics demo image
    return "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"


def test_inference(inferencer, image):
    results = inferencer.predict(image)
    if len(results[0]) == 0:
        warnings.warn(f"No objects were detected in the test image {image}")


@pytest.mark.parametrize("threshold", [0.8, 0.9, 1.0])
def test_nms_confidence_threshold(inferencer, image, threshold):
    results = inferencer.predict(image, conf=threshold)
    assert all(conf >= threshold for *_, conf in results[0])
