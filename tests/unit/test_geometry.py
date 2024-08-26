from htrflow.utils import geometry


def test_point_unpacking():
    p = geometry.Point(1, 2)
    x, y = p
    assert x == p.x
    assert y == p.y


def test_point_indexing():
    p = geometry.Point(1, 2)
    assert p[0] == 1
    assert p[1] == 2


def test_point_init():
    p1 = geometry.Point(1, 1)
    p2 = geometry.Point(*p1)
    assert p1 == p2


def test_point_move():
    p1 = geometry.Point(0, 0)
    p2 = p1.move((1, 1))
    assert p2 == geometry.Point(1, 1)


def test_point_move_immutable():
    p = geometry.Point(0, 0)
    p.move((1, 1))
    assert p == geometry.Point(0, 0)


def test_bbox_height_width():
    bbox = geometry.Bbox(0, 0, 10, 10)
    assert bbox.height == 10
    assert bbox.width == 10


def test_bbox_xywh():
    bbox = geometry.Bbox(1, 1, 11, 21)
    assert bbox.xywh == (1, 1, 10, 20)


def test_bbox_xyxy():
    bbox = (0, 0, 10, 20)
    assert geometry.Bbox(*bbox).xyxy == bbox


def test_bbox_xxyy():
    bbox = geometry.Bbox(0, 0, 10, 20)
    assert bbox.xxyy == (0, 10, 0, 20)


def test_bbox_points():
    p1 = geometry.Point(10, 20)
    p2 = geometry.Point(30, 40)
    bbox = geometry.Bbox(*p1, *p2)
    assert bbox.p1 == p1
    assert bbox.p2 == p2


def test_bbox_center():
    bbox = geometry.Bbox(10, 10, 30, 40)
    assert bbox.center.x == 20
    assert bbox.center.y == 25


def test_bbox_polygon_conversion():
    bbox = geometry.Bbox(10, 20, 30, 40)
    assert bbox.polygon().bbox() == bbox


def test_bbox_move():
    bbox1 = geometry.Bbox(0, 0, 10, 10)
    to = geometry.Point(10, 10)
    bbox2 = bbox1.move(to)

    assert bbox1.height == bbox2.height
    assert bbox1.width == bbox2.width
    assert bbox2.p1 == to


def test_bbox_move_immutable():
    bbox1 = geometry.Bbox(0, 0, 10, 10)
    to = geometry.Point(10, 10)
    bbox2 = bbox1.move(to)
    assert bbox1 != bbox2


def test_bbox_iter():
    data = (0, 0, 10, 10)
    assert all(i == j for i, j in zip(geometry.Bbox(*data), data))


def test_bbox_getitem():
    bbox = geometry.Bbox(0, 0, 10, 10)
    assert bbox[0] == 0
    assert bbox[1] == 0
    assert bbox[2] == 10
    assert bbox[3] == 10


def test_polygon_iter():
    points = [geometry.Point(i, i) for i in range(5)]
    polygon = geometry.Polygon(points)
    assert all(p1 == p2 for p1, p2 in zip(points, polygon))


def test_polygon_getitem():
    point = geometry.Point(0, 0)
    polygon = geometry.Polygon([point])
    assert point == polygon[0]


def test_polygon_from_tuples():
    pol_tuples = geometry.Polygon([(0, 0)])
    pol_points = geometry.Polygon([geometry.Point(0, 0)])
    assert pol_points[0] == pol_tuples[0]


def test_polygon_move():
    n_points = 5
    dest = (5, 5)

    # Create a polygon with points (0,0), (1,1) ...
    points = [geometry.Point(i, i) for i in range(n_points)]
    polygon = geometry.Polygon(points)
    polygon = polygon.move(dest)

    # We expect a new polygon (5,5), (6,6) ...
    exptected_points = [geometry.Point(i + dest[0], i + dest[1]) for i in range(n_points)]

    assert all(point == expected_point for point, expected_point in zip(polygon, exptected_points))


def test_polygon_array():
    n_points = 5
    points = [geometry.Point(i, i) for i in range(n_points)]
    polygon = geometry.Polygon(points)
    assert all(p1.x == p2[0] and p1.y == p2[1] for p1, p2 in zip(points, polygon.as_nparray()))
