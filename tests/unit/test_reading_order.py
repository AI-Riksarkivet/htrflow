from htrflow.postprocess import reading_order
from htrflow.utils.geometry import Bbox


def test_lrtd_one_line():
    # Case: 20 boxes with increasing x coordinate and slight variation
    # in y coordinate (0 or 1). These should be considered to be on the
    # same line and thus only the x coordinate matters.
    nx, ny = 10, 2
    bboxes = [Bbox(x, y, x + 10, y + 10) for x in range(nx) for y in range(ny)]
    assert reading_order.left_right_top_down(bboxes) == list(range(nx * ny))


def test_lrtd_two_lines():
    # Case: 20 boxes with increasing x coordinate and significant
    # variation in y axis (0 or 20). These should be considered to be
    # two lines.
    nx, ny = 10, 2
    bboxes = [Bbox(x, y * 20, x + 10, y + 10) for x in range(nx) for y in range(ny)]
    expected_order = [n * 2 for n in range(nx)] + [n * 2 + 1 for n in range(nx)]
    assert reading_order.left_right_top_down(bboxes) == expected_order
