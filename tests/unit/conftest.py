import random

import pytest

from htrflow.document import Document, Region, Text
from htrflow.utils.geometry import Polygon


@pytest.fixture
def demo_image():
    return "examples/images/pages/A0068699_00021.jpg"


@pytest.fixture
def document_with_regions_and_transcribed_lines(demo_image):
    document = Document(demo_image)
    n_regions = 3
    n_lines = 10
    for i in range(n_regions):
        polygon = dummy_polygon(1000, 1000)
        region = Region(polygon)
        region.attach(document)
        for j in range(n_lines):
            polygon = dummy_polygon(100, 100)
            text = Text(text=f"Transcription of line {j} in region {i}", confidence=random.random())
            Region(polygon, transcription=[text]).attach(region)
    return document


def dummy_polygon(height, width):
    return Polygon([(0, 0), (0, height), (width, height), (width, 0)])
