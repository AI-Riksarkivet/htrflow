import xml.etree.ElementTree as ET

import pytest

from htrflow.document import Document, Region, Text
from htrflow.serialization import AltoXML, PageXML, Serializer
from htrflow.utils.geometry import Bbox


@pytest.fixture
def alto_namespace():
    return {"": "http://www.loc.gov/standards/alto/ns-v4#"}


@pytest.fixture
def document():
    demo_image = "examples/images/pages/A0068699_00021.jpg"
    document = Document(demo_image)
    hpos = vpos = 100
    height = width = 100

    region_polygon = Bbox(0, 0, width, height).move((hpos, vpos)).polygon()
    line_bbox = Bbox(0, 0, int(width * 0.5), int(height * 0.5))

    # Define structure of testing document
    n_lines = 10
    n_regions = 3
    for j in range(n_regions):
        Region(
            polygon=region_polygon,
            regions=[
                Region(
                    polygon=line_bbox.move((i, i)).polygon(),
                    transcription=[Text(f"transcription of line {i} of region {j}")],
                )
                for i in range(n_lines)
            ],
        ).attach(document)

    return document


@pytest.mark.parametrize(
    "serializer",
    [AltoXML(), PageXML(template_name="page2013"), PageXML(template_name="page2019")],
    ids=["alto", "page2013", "page2019"],
)
def test_xml_validates(document, serializer: Serializer):
    doc = serializer.serialize(document)
    serializer.validate(doc)


@pytest.mark.parametrize(
    "serializer",
    [AltoXML(), PageXML(template_name="page2013"), PageXML(template_name="page2019")],
    ids=["alto", "page2013", "page2019"],
)
def test_xml_line_missing_transcription(document: Document, serializer: Serializer):
    document.regions[0].regions[0].transcription = []
    doc = serializer.serialize(document)
    serializer.validate(doc)


@pytest.mark.parametrize(
    "serializer",
    [AltoXML(), PageXML(template_name="page2013"), PageXML(template_name="page2019")],
    ids=["alto", "page2013", "page2019"],
)
def test_xml_region_missing_transcription(document: Document, serializer: Serializer):
    document.regions[0].regions = []
    doc = serializer.serialize(document)
    serializer.validate(doc)


@pytest.mark.parametrize(
    "serializer",
    [AltoXML(), PageXML(template_name="page2013"), PageXML(template_name="page2019")],
    ids=["alto", "page2013", "page2019"],
)
def test_xml_special_characters(document: Document, serializer: Serializer):
    a_very_special_string = "åäöÅÄÖ&<>'" + '"'
    document.regions[0].regions[0].transcription[0].text = a_very_special_string
    doc = serializer.serialize(document)
    serializer.validate(doc)


@pytest.mark.parametrize(
    "serializer",
    [AltoXML(), PageXML(template_name="page2013"), PageXML(template_name="page2019")],
    ids=["alto", "page2013", "page2019"],
)
def test_xml_no_content(document: Document, serializer: Serializer):
    document.regions = []
    doc = serializer.serialize(document)
    serializer.validate(doc)


@pytest.mark.parametrize(
    "serializer",
    [AltoXML(), PageXML(template_name="page2013"), PageXML(template_name="page2019")],
    ids=["alto", "page2013", "page2019"],
)
def test_xml_word_level(document: Document, serializer: Serializer):
    document.regions = [Region(Bbox(0, 0, 1000, 1000).polygon(), regions=document.regions)]
    doc = serializer.serialize(document)
    serializer.validate(doc)


def test_alto_correct_image_info(document: Document, alto_namespace: dict):
    serializer = AltoXML()
    doc = serializer.serialize(document)
    doc = ET.fromstring(doc)

    filename = doc.find(".//fileName", alto_namespace).text
    height = int(doc.find(".//Page", alto_namespace).attrib.get("HEIGHT"))
    width = int(doc.find(".//Page", alto_namespace).attrib.get("WIDTH"))

    assert filename == "A0068699_00021"
    assert height == document.image.height
    assert width == document.image.width


def test_alto_nested_coordinates(document: Document, alto_namespace: dict):
    serializer = AltoXML()
    doc = serializer.serialize(document)
    doc = ET.fromstring(doc)

    for block in doc.findall(".//TextBlock", alto_namespace):
        hpos = int(block.attrib.get("HPOS"))
        vpos = int(block.attrib.get("VPOS"))

        for i, line in enumerate(block.findall("./TextLine", alto_namespace)):
            assert hpos + i == int(line.attrib.get("HPOS"))
            assert vpos + i == int(line.attrib.get("VPOS"))
