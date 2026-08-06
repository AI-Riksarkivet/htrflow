import pytest
import torch
import torch.nn.functional as F
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
    """
    Sanity test for WordLevelTrOCR.
    """
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

    # Check that words don't contain whitespace
    for word in words:
        assert not any(ch.isspace() for ch in word.transcription[0].text)


def test_estimate_bbox_from_empty_attention_heatmap():
    """
    Tests that `estimate_bbox_from_attention_heatmap` returns None if
    the given heatmap is empty.
    """
    heatmap_size = (12, 64)
    image_size = (120, 640)
    heatmap = torch.zeros(heatmap_size)
    assert trocr.estimate_bbox_from_attention_heatmap(heatmap, image_size) is None


def test_estimate_bbox_from_constant_attention_heatmap():
    """
    Tests that `estimate_bbox_from_attention_heatmap` returns an as
    large as possible bounding box when the attention heatmap is
    constant.
    """
    heatmap_size = (12, 64)
    image_size = (120, 640)
    heatmap = torch.ones(heatmap_size)
    bbox = trocr.estimate_bbox_from_attention_heatmap(heatmap, image_size)
    assert bbox.height, bbox.width == image_size


def test_estimate_bbox_from_tiny_attention_heatmap():
    """
    Tests that `estimate_bbox_from_attention_heatmap` can handle
    heatmaps of surprising shapes.
    """
    heatmap_size = (1, 1)
    image_size = (120, 640)
    heatmap = torch.ones(heatmap_size)
    bbox = trocr.estimate_bbox_from_attention_heatmap(heatmap, image_size)
    assert bbox.height, bbox.width == image_size


def test_estimate_bbox_from_attention_heatmap_size_invariant():
    """
    Tests that `estimate_bbox_from_attention_heatmap` returns simliar
    results for heatmaps at different shapes and scales.
    """

    image_size = (120, 640)
    abs_tol = 5  # Allow up to 5 pixels diff in bbox coordinates

    heatmap = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    sizes = [(6, 6), (6, 12), (12, 12), (12, 64), (64, 12)]

    bboxes = []
    for size in sizes:
        heatmap_resized = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size)
        heatmap_resized = heatmap_resized.squeeze(0).squeeze(0)
        bbox = trocr.estimate_bbox_from_attention_heatmap(heatmap_resized, image_size)
        bboxes.append(bbox)

    assert all(bboxes[0].height == pytest.approx(bbox.height, abs=abs_tol) for bbox in bboxes)
    assert all(bboxes[0].width == pytest.approx(bbox.width, abs=abs_tol) for bbox in bboxes)
    assert all(bboxes[0].xmin == pytest.approx(bbox.xmin, abs=abs_tol) for bbox in bboxes)
    assert all(bboxes[0].ymin == pytest.approx(bbox.ymin, abs=abs_tol) for bbox in bboxes)
