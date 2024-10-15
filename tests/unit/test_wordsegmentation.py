import pytest

from htrflow.models.huggingface.trocr import WordLevelTrOCR
from htrflow.utils.imgproc import read


testdata = [
    ("examples/images/lines/A0068699_00021_region0_line1.jpg", [0, 250, 1000, 1530, 1785]),
    ("examples/images/lines/A0068699_00021_region0_line2.jpg", [0, 680, 1000, 1525]),
    ("examples/images/lines/A0068699_00021_region0_line3.jpg", [0, 250, 1000, 1325, 1830]),
    ("examples/images/lines/A0068699_00021_region0_line4.jpg", [0, 520, 1080, 1845]),
]


@pytest.fixture(scope="module")
def model():
    return WordLevelTrOCR("Riksarkivet/trocr-base-handwritten-hist-swe-2")


@pytest.mark.parametrize("image,expected_segmentation", testdata)
def test_wordsegmentation(model, image, expected_segmentation):
    image = read(image)
    results = model([image])
    result = results[0]
    segmentation = [bbox.xmin for bbox in result.bboxes]
    assert segmentation == pytest.approx(expected_segmentation, abs=100)
