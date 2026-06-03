import pytest
from PIL import Image

from htrflow.document import Region


pytest.importorskip("transformers", reason="Transformers dependencies not installed")


from htrflow.models.huggingface import trocr


# Some test data.
# The coordinates represent the mid point of each word.
testdata = [
    ("examples/images/lines/A0068699_00021_region0_line0.jpg", [(421, 105), (583, 86), (729, 105)]),
    (
        "examples/images/lines/A0068699_00021_region0_line1.jpg",
        [(156, 121), (611, 114), (1248, 114), (1678, 117), (1931, 108)],
    ),
    ("examples/images/lines/A0068699_00021_region0_line2.jpg", [(348, 135), (851, 133), (1274, 135), (1781, 122)]),
    (
        "examples/images/lines/A0068699_00021_region0_line3.jpg",
        [(110, 122), (659, 125), (1188, 121), (1626, 115), (1926, 111)],
    ),
    ("examples/images/lines/A0068699_00021_region0_line4.jpg", [(293, 119), (805, 121), (1482, 111), (1950, 105)]),
]


@pytest.fixture(scope="module")
def model():
    return trocr.WordLevelTrOCR(model="Riksarkivet/trocr-base-handwritten-hist-swe-2")


@pytest.mark.parametrize("image,expected_midpoints", testdata)
def test_wordsegmentation(model, image, expected_midpoints):
    image = Image.open(image)
    results = model([image])
    result = results[0]
    words = [region for region in result if isinstance(region, Region)]

    assert len(words) == len(expected_midpoints)

    # Check that the midpoints are covered by the returned boxes
    for word, (x, y) in zip(words, expected_midpoints):
        bbox = word.polygon.bbox
        assert bbox.xmin <= x <= bbox.xmax
        assert bbox.ymin <= y <= bbox.ymax
