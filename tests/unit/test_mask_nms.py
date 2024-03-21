import numpy as np
import pytest

# from htrflow_core.postprocess.mask_nms import multiclass_mask_nms, mask_nms
from htrflow_core.results import Result, Segment


@pytest.fixture
def results_with_mask():
    orig_shape = (200, 200)
    image = np.zeros(orig_shape, dtype=np.uint8)

    mask_a = np.zeros(orig_shape, dtype=np.uint8)
    mask_a[20:70, 20:70] = 1
    mask_b = np.zeros(orig_shape, dtype=np.uint8)
    mask_b[10:180, 10:180] = 1
    mask_c = np.zeros(orig_shape, dtype=np.uint8)
    mask_c[30:70, 30:70] = 1

    segment_a = Segment(mask=mask_a, class_label="class_1", orig_shape=orig_shape)
    segment_b = Segment(mask=mask_b, class_label="class_2", orig_shape=orig_shape)
    segment_c = Segment(mask=mask_c, class_label="class_1", orig_shape=orig_shape)

    return Result(image=image, metadata={}, segments=[segment_a, segment_b, segment_c])


def test_multiclass_mask_nms(results_with_mask):
    # TODO: test work for different labels
    pass


def test_mask_nms(results_with_mask):
    # TODO
    pass
