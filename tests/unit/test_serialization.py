import pytest

from htrflow_core import serialization
from htrflow_core.dummies.dummy_models import RecognitionModel, SegmentationModel
from htrflow_core.results import RecognizedText
from htrflow_core.volume import volume


@pytest.fixture
def demo_image():
    return "data/demo_images/demo_image.jpg"


@pytest.fixture
def alto():
    return serialization.AltoXML()


@pytest.fixture
def page():
    return serialization.PageXML()


@pytest.fixture
def demo_page_unsegmented(demo_image):
    node = volume.PageNode(demo_image)
    model = RecognitionModel()
    result = model([node.image])
    node.add_data(**result[0].data[0])
    return node


@pytest.fixture
def demo_page_segmented_once(demo_image):
    node = volume.PageNode(demo_image)
    model = SegmentationModel()
    results = model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        node.segment(result.segments)
    model = RecognitionModel()
    results = model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        leaf.add_data(**result.data[0])
    return node


@pytest.fixture
def demo_page_segmented_twice(demo_image):
    node = volume.PageNode(demo_image)
    model = SegmentationModel()
    for _ in range(2):
        results = model(node.segments())
        for result, leaf in zip(results, node.leaves()):
            leaf.segment(result.segments)
    model = RecognitionModel()
    results = model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        leaf.add_data(**result.data[0])
    return node


@pytest.fixture
def demo_page_segmented_thrice(demo_image):
    node = volume.PageNode(demo_image)
    model = SegmentationModel()
    for _ in range(3):
        results = model(node.segments())
        for result, leaf in zip(results, node.leaves()):
            leaf.segment(result.segments)
    model = RecognitionModel()
    results = model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        leaf.add_data(**result.data[0])
    return node


def test_alto_unsegmented(demo_page_unsegmented, alto):
    doc = alto.serialize(demo_page_unsegmented)
    assert doc is None


def test_alto_segmented(demo_page_segmented_once, alto):
    doc = alto.serialize(demo_page_segmented_once)
    alto.validate(doc)


def test_alto_segmented_twice(demo_page_segmented_twice, alto):
    doc = alto.serialize(demo_page_segmented_twice)
    alto.validate(doc)


def test_alto_segmented_thrice(demo_page_segmented_thrice, alto):
    doc = alto.serialize(demo_page_segmented_thrice)
    alto.validate(doc)


def test_alto_escape_characters(demo_page_segmented_thrice, alto):
    node, *_ = demo_page_segmented_thrice.leaves()
    to_be_escaped = "\"'&<>"  # these characters may not appear in the xml
    node.add_data(text_result=RecognizedText([to_be_escaped], [1]))
    doc = alto.serialize(demo_page_segmented_thrice)
    alto.validate(doc)


def test_page_unsegmented(demo_page_unsegmented, page):
    doc = page.serialize(demo_page_unsegmented)
    assert doc is None


def test_page_segmented(demo_page_segmented_once, page):
    doc = page.serialize(demo_page_segmented_once)
    page.validate(doc)


def test_page_segmented_twice(demo_page_segmented_twice, page):
    doc = page.serialize(demo_page_segmented_twice)
    page.validate(doc)


def test_page_segmented_thrice(demo_page_segmented_thrice, page):
    doc = page.serialize(demo_page_segmented_thrice)
    page.validate(doc)
