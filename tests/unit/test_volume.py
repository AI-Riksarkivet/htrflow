import pickle

import pytest

from htrflow_core.dummies.dummy_models import RecognitionModel, SegmentationModel
from htrflow_core.volume import node, postprocess, volume


@pytest.fixture
def demo_image():
    return "data/demo_images/demo_image.jpg"


@pytest.fixture
def demo_volume_unsegmented(demo_image):
    n_images = 5
    vol = volume.Volume([demo_image] * n_images)
    return vol


@pytest.fixture
def demo_volume_segmented(demo_image):
    n_images = 5
    vol = volume.Volume([demo_image] * n_images)
    model = SegmentationModel()
    result = model(vol.images())
    vol.update(result)
    return vol


@pytest.fixture
def demo_volume_segmented_nested(demo_image):
    n_images = 5
    vol = volume.Volume([demo_image] * n_images)
    model = SegmentationModel()
    result = model(vol.images())
    vol.update(result)
    result = model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_volume_segmented_nested_with_text(demo_image):
    n_images = 5
    vol = volume.Volume([demo_image] * n_images)
    model = SegmentationModel()
    result = model(vol.images())
    vol.update(result)
    result = model(vol.segments())
    vol.update(result)
    model = RecognitionModel()
    result = model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_volume_with_text(demo_image):
    n_images = 1
    vol = volume.Volume([demo_image] * n_images)
    model = RecognitionModel()
    result = model(vol.images())
    vol.update(result)
    return vol


def one_layer_tree(n_children=3):
    root = node.Node()
    root.children = [node.Node(root) for _ in range(n_children)]
    return root


def two_layer_tree(n_children=3):
    root = node.Node()
    root.children = [node.Node(root) for _ in range(n_children)]
    for child in root.children:
        child.children = [node.Node(child) for _ in range(n_children)]
    return root


def test_node_getitem():
    root = one_layer_tree()
    assert root[0] == root.children[0]


def test_node_getitem_grandchild():
    root = two_layer_tree()
    assert root[0, 0] == root.children[0].children[0]


def test_leaves():
    root = one_layer_tree()
    assert {*root.leaves()} == {*root.children}


def test_leaves_grandchildren():
    root = two_layer_tree()
    assert {*root.leaves()} == {grandchild for child in root.children for grandchild in child.children}


def test_traverse():
    root = one_layer_tree()
    assert {*root.traverse()} == {root, *root.children}


def test_traverse_grandchildren():
    root = two_layer_tree()
    children = {*root.children}
    grandchildren = {grandchild for child in children for grandchild in child.children}
    assert {*root.traverse()} == {root, *children, *grandchildren}


def test_node_detach():
    n_children = 3
    root = two_layer_tree(n_children)
    root[0].detach()
    assert len(root.children) == n_children - 1


def test_node_prune():
    n_children = 3
    root = two_layer_tree(n_children)

    def filter_(node):
        return node.depth() > 1

    root.prune(filter_)
    assert not any(filter_(node) for node in root.traverse())


def test_update_wrong_type(demo_image):
    root = volume.Volume([demo_image])
    with pytest.raises(TypeError) as _:
        root.update(5)


def test_update_segmentation(demo_image):
    segmentation_model = SegmentationModel("mask")
    node = volume.PageNode(demo_image)
    result, *_ = segmentation_model([node.image])
    node.create_segments(result.segments)
    assert len(node.children) == len(result.segments)


def test_update_segmentation_bbox(demo_image):
    segmentation_model = SegmentationModel("mask")
    node = volume.PageNode(demo_image)
    result, *_ = segmentation_model([node.image])
    node.segment(result.segments)
    segment_index = 0
    segment_bbox = result.segments[segment_index].bbox
    node_bbox = node[segment_index].get("segment").bbox
    assert node_bbox == segment_bbox


def test_update_segmentation_width_height(demo_image):
    segmentation_model = SegmentationModel("mask")
    node = volume.PageNode(demo_image)
    result, *_ = segmentation_model([node.image])
    node.create_segments(result.segments)
    segment_index = 0
    x1, y1, x2, y2 = result.segments[segment_index].bbox
    assert x2 - x1 == node[segment_index].width
    assert y2 - y1 == node[segment_index].height


def test_update_nested_segmentation_coordinates(demo_volume_segmented_nested):
    page = demo_volume_segmented_nested[0]
    segment = page[0]
    nested_segment = page[0, 0]
    parent_x = segment.coord.x
    nested_segment_x_relative_to_parent = nested_segment.segment.bbox[0]
    assert nested_segment.coord.x == parent_x + nested_segment_x_relative_to_parent

    parent_y = segment.coord.y
    nested_segment = page[0, 0]
    nested_segment_y_relative_to_parent = nested_segment.segment.bbox[1]
    assert nested_segment.coord.y == parent_y + nested_segment_y_relative_to_parent


def test_update_region_text(demo_volume_segmented):
    model = RecognitionModel()
    result = model(demo_volume_segmented.segments())
    demo_volume_segmented.update(result)
    page = demo_volume_segmented[0]
    node = page[0]
    assert node.get("text_result") == result[0].data[0].get("text_result")


def test_polygon_nested(demo_volume_segmented_nested):
    page = demo_volume_segmented_nested[0]
    node = page[0]
    nested_node = page[0, 0]
    # .polygon attribute should be relative to original image
    # but segment.polygon should be relative to parent
    expected_polygon = [(node.coord.x + x, node.coord.y + y) for x, y in nested_node.segment.polygon]
    assert all(p1[0] == p2[0] for p1, p2 in zip(nested_node.polygon, expected_polygon))


def test_polygon_not_nested(demo_volume_segmented):
    ...
    # page = demo_volume_segmented[0]
    # node = page[0]
    # TODO fix so that the types match
    # assert node.polygon == node.segment.polygon


def test_polygon_image(demo_volume_unsegmented):
    page = demo_volume_unsegmented[0]
    xs = [x for x, _ in page.polygon]
    ys = [y for _, y in page.polygon]
    # Page polygon should cover the entire page
    assert min(xs) == 0
    assert max(xs) == page.width
    assert min(ys) == 0
    assert max(ys) == page.height


def test_volume_update_wrong_size(demo_volume_segmented):
    with pytest.raises(ValueError) as _:
        demo_volume_segmented.update([])


def test_volume_iter(demo_volume_segmented):
    # iterating over the volume should iterate over its children (pages)
    assert all(a == b for a, b in zip(demo_volume_segmented, demo_volume_segmented.children))


def test_volume_segments_depth_none(demo_volume_segmented):
    leaves = demo_volume_segmented.leaves()
    segments = demo_volume_segmented.segments()
    # (A==B).all() checks if two arrays are equal (cannot do A==B)
    assert all((img == leaf.image).all() for img, leaf in zip(segments, leaves))


# Tests of volume.save()
# More thorough serialization testing is done in test_seralization
def test_volume_save_text(tmpdir, demo_volume_with_text):
    demo_volume_with_text.save(tmpdir, "txt")
    assert len(tmpdir.listdir()) == 1


def test_pickling(demo_volume_segmented_nested):
    # TODO: there is probably a better way of ensuring that the pickled
    # volume is restored properly. Here I use an arbitrary demo volume
    # and check that some attributes are equal.
    picklestring = pickle.dumps(demo_volume_segmented_nested)
    vol = pickle.loads(picklestring)
    assert isinstance(vol, volume.Volume)  # sanity check
    assert all(p1 == p2 for p1, p2 in zip(vol[0, 0].polygon[0], demo_volume_segmented_nested[0, 0].polygon[0]))
    assert vol[0, 0, 0].label == demo_volume_segmented_nested[0, 0, 0].label


def test_save_and_load_pickle(tmpdir, demo_volume_segmented_nested):
    # TODO: see test_pickling
    picklefile = demo_volume_segmented_nested.pickle(tmpdir)
    vol = volume.Volume.from_pickle(picklefile)
    assert vol[0, 0].height == demo_volume_segmented_nested[0, 0].height
    assert vol[0, 0].parent.width == demo_volume_segmented_nested[0].width


@pytest.mark.parametrize("threshold", [0.3, 0.6, 0.9, 0.99])
def test_remove_noise_regions(demo_volume_segmented_nested_with_text, threshold):
    pruned = postprocess.remove_noise_regions(demo_volume_segmented_nested_with_text, threshold)
    assert not any(postprocess.is_noise(node, threshold) for node in pruned.traverse())
