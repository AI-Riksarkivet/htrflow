import pytest

from htrflow import serialization
from htrflow.results import RecognizedText


@pytest.fixture
def alto():
    return serialization.AltoXML()


@pytest.fixture
def page():
    return serialization.PageXML()


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
