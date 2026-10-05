import random

import pytest
import torch
from ultralytics.engine.results import Masks, Results

from htrflow.models.ultralytics.yolo import _postprocess


random.seed(0)
torch.random.manual_seed(0)


def dummy_results():
    """
    A dummy `Results` object with randomized data, as returned by YOLO
    models, to use for testing.
    """

    h, w = 100, 100
    n_detections = random.randint(1, 10)

    masks = torch.zeros((n_detections, h, w), dtype=torch.bool)
    boxes = []
    for n in range(n_detections):
        x1 = random.randint(0, w // 2)
        x2 = random.randint(x1, w)
        y1 = random.randint(0, h // 2)
        y2 = random.randint(y1, h)
        masks[n, y1:y2, x1:x2] = True
        conf = 0.99
        label = 0
        boxes.append((x1, y1, x2, y2, conf, label))

    boxes = torch.tensor(boxes, dtype=torch.uint8)
    orig_img = torch.rand(h, w)
    return Results(orig_img, "", names={0: "label"}, boxes=boxes, masks=masks)


@pytest.mark.parametrize("_", range(10))
def test_postprocess(_):
    """
    Sanity check: Randomized results should be postprocessed without raising
    any exceptions.
    """
    _postprocess(dummy_results(), 0.01)


def test_postprocess_prioritize_masks():
    """
    Tests that `postprocess` prioritizes masks: If both bounding boxes
    and masks are given, the Region's polygon should be derived from the
    masks.
    """

    # Get a dummy result and its postprocessed regions.
    results = dummy_results()
    regions_a = _postprocess(results, 0.01)

    # Set the boxes of the dummy result to zero. This should not affect
    # the postprocessed regions, since they are based on the masks.
    results.boxes.data = torch.zeros_like(results.boxes.data)
    regions_b = _postprocess(results, 0.01)

    # Check that the two versions are equal.
    for a, b in zip(regions_a, regions_b):
        assert (a.polygon.as_nparray() == b.polygon.as_nparray()).all()


def test_postprocess_no_masks():
    """
    Tests that postprocess is able to handle results without masks.
    """

    # Postprocess results with masks.
    results = dummy_results()
    regions_a = _postprocess(results, 0.01)

    # Set masks to None and postprocess again.
    results.masks = None
    regions_b = _postprocess(results, 0.01)

    # Check that the two versions are roughly equal.
    for a, b in zip(regions_a, regions_b):
        assert a.polygon.bbox.xyxy == pytest.approx(b.polygon.bbox.xyxy, abs=1)


def test_postprocess_bad_masks():
    """
    Tests that postprocess is able to handle results with bad masks.
    """

    # Postprocess results with masks.
    results = dummy_results()
    regions_a = _postprocess(results, 0.01)

    # Set masks to zeros and postprocess again. Bad masks (here: empty masks)
    # should not cause any issues in postprocess, and the bounding box fallback
    # should be activated.
    results.masks = Masks(torch.zeros(results.masks.data.shape), results.orig_shape)
    regions_b = _postprocess(results, 0.01)

    # Check that the two versions are roughly equal. This means that the
    # bounding box fallback was activated correctly.
    for a, b in zip(regions_a, regions_b):
        assert a.polygon.bbox.xyxy == pytest.approx(b.polygon.bbox.xyxy, abs=1)
