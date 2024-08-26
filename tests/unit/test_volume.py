import pickle

import pytest

from htrflow.volume import node, volume


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
    root = volume.Collection([demo_image])
    with pytest.raises(TypeError) as _:
        root.update(5)


def test_update_nested_segmentation_coordinates(demo_collection_segmented_nested):
    page = demo_collection_segmented_nested[0]
    segment = page[0]
    nested_segment = page[0, 0]
    parent_x = segment.coord.x
    nested_segment_x_relative_to_parent = nested_segment.segment.bbox[0]
    assert nested_segment.coord.x == parent_x + nested_segment_x_relative_to_parent

    parent_y = segment.coord.y
    nested_segment = page[0, 0]
    nested_segment_y_relative_to_parent = nested_segment.segment.bbox[1]
    assert nested_segment.coord.y == parent_y + nested_segment_y_relative_to_parent


def test_polygon_nested(demo_collection_segmented_nested):
    page = demo_collection_segmented_nested[0]
    node = page[0]
    nested_node = page[0, 0]
    # .polygon attribute should be relative to original image
    # but segment.polygon should be relative to parent
    expected_polygon = [(node.coord.x + x, node.coord.y + y) for x, y in nested_node.segment.polygon]
    assert all(p1[0] == p2[0] for p1, p2 in zip(nested_node.polygon, expected_polygon))


def test_polygon_not_nested(demo_collection_segmented):
    ...
    # page = demo_collection_segmented[0]
    # node = page[0]
    # TODO fix so that the types match
    # assert node.polygon == node.segment.polygon


def test_polygon_image(demo_collection_unsegmented):
    page = demo_collection_unsegmented[0]
    xs = [x for x, _ in page.polygon]
    ys = [y for _, y in page.polygon]
    # Page polygon should cover the entire page
    assert min(xs) == 0
    assert max(xs) == page.width
    assert min(ys) == 0
    assert max(ys) == page.height


def test_collection_update_wrong_size(demo_collection_segmented):
    with pytest.raises(ValueError) as _:
        demo_collection_segmented.update([])


def test_collection_iter(demo_collection_segmented):
    # iterating over the volume should iterate over its children (pages)
    assert all(a == b for a, b in zip(demo_collection_segmented, demo_collection_segmented.pages))


def test_collection_segments_depth_none(demo_collection_segmented):
    leaves = demo_collection_segmented.leaves()
    segments = demo_collection_segmented.segments()
    # (A==B).all() checks if two arrays are equal (cannot do A==B)
    assert all((img == leaf.image).all() for img, leaf in zip(segments, leaves))


# Tests of volume.save()
# More thorough serialization testing is done in test_seralization
def test_collection_save_text(tmpdir, demo_collection_with_text):
    demo_collection_with_text.save(tmpdir, "txt")
    assert len(tmpdir.listdir()) == 1


def test_pickling(demo_collection_segmented_nested):
    # TODO: there is probably a better way of ensuring that the pickled
    # volume is restored properly. Here I use an arbitrary demo volume
    # and check that some attributes are equal.
    picklestring = pickle.dumps(demo_collection_segmented_nested)
    vol = pickle.loads(picklestring)
    assert isinstance(vol, volume.Collection)  # sanity check
    assert all(p1 == p2 for p1, p2 in zip(vol[0, 0].polygon[0], demo_collection_segmented_nested[0, 0].polygon[0]))
    assert vol[0, 0, 0].label == demo_collection_segmented_nested[0, 0, 0].label
