import pytest

import htrflow_core.serialization as serialization
import htrflow_core.volume as volume
from htrflow_core.dummies.dummy_models import RecognitionModel, SegmentationModel


@pytest.fixture
def demo_image():
    return "data/demo_image.jpg"


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
    node.update(result[0])
    return node


@pytest.fixture
def demo_page_segmented_once(demo_image):
    node = volume.PageNode(demo_image)
    model = SegmentationModel()
    result = model([node.image])
    node.update(result[0])
    model = RecognitionModel()
    result = model([node.image for node in node.leaves()])
    node.update(result[0])
    return node


@pytest.fixture
def demo_page_segmented_twice(demo_image):
    node = volume.PageNode(demo_image)
    model = SegmentationModel()
    result = model([node.image])
    node.update(result[0])
    model = RecognitionModel()
    for _ in range(2):
        result = model([node.image for node in node.leaves()])
        node.update(result[0])
    return node


@pytest.fixture
def demo_page_segmented_thrice(demo_image):
    node = volume.PageNode(demo_image)
    model = SegmentationModel()
    result = model([node.image])
    node.update(result[0])
    model = RecognitionModel()
    for _ in range(3):
        result = model([node.image for node in node.leaves()])
        node.update(result[0])
    return node


def test_alto_unsegmented(demo_page_unsegmented, alto):
    doc = alto.serialize(demo_page_unsegmented)
    alto.validate(doc)


def test_alto_segmented(demo_page_segmented_once, alto):
    doc = alto.serialize(demo_page_segmented_once)
    alto.validate(doc)


def test_alto_segmented_twice(demo_page_segmented_twice, alto):
    doc = alto.serialize(demo_page_segmented_twice)
    alto.validate(doc)


def test_alto_segmented_thrice(demo_page_segmented_thrice, alto):
    doc = alto.serialize(demo_page_segmented_thrice)
    alto.validate(doc)


def test_page_unsegmented(demo_page_unsegmented, page):
    doc = page.serialize(demo_page_unsegmented)
    page.validate(doc)


def test_page_segmented(demo_page_segmented_once, page):
    doc = page.serialize(demo_page_segmented_once)
    page.validate(doc)


def test_page_segmented_twice(demo_page_segmented_twice, page):
    doc = page.serialize(demo_page_segmented_twice)
    page.validate(doc)


def test_page_segmented_thrice(demo_page_segmented_thrice, page):
    doc = page.serialize(demo_page_segmented_thrice)
    page.validate(doc)
