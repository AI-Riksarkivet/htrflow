import pytest

from htrflow import serialization
from htrflow.document import Text


@pytest.fixture
def alto():
    return serialization.AltoXML()


@pytest.fixture
def page():
    return serialization.PageXML()


def test_alto_regions_and_lines(document_with_regions_and_transcribed_lines, alto):
    doc = alto.serialize(document_with_regions_and_transcribed_lines)
    alto.validate(doc)


def test_alto_escape_characters(document_with_regions_and_transcribed_lines, alto):
    node, *_ = document_with_regions_and_transcribed_lines.leaves()
    to_be_escaped = "\"'&<>"  # these characters may not appear in the xml
    text = Text(to_be_escaped)
    text.attach(node)
    doc = alto.serialize(document_with_regions_and_transcribed_lines)
    alto.validate(doc)


def test_page_segmented_twice(document_with_regions_and_transcribed_lines, page):
    doc = page.serialize(document_with_regions_and_transcribed_lines)
    page.validate(doc)
