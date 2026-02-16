import random
from string import ascii_letters

import pytest

from htrflow.document import Collection, Document, Region, Text
from htrflow.utils.geometry import Bbox


@pytest.fixture
def demo_image():
    return "examples/images/pages/A0068699_00021.jpg"


@pytest.fixture
def demo_page_unsegmented(demo_image):
    document = Document(demo_image)
    result = dummy_text_recognition_model([document.image])
    for text in result[0]:
        text.attach(document)
    return document


@pytest.fixture
def demo_page_segmented_once(demo_image):
    document = Document(demo_image)
    results = dummy_segmentation_model(document.segments())
    for result, leaf in zip(results, document.leaves()):
        for item in result:
            item.attach(leaf)
    results = dummy_text_recognition_model(document.segments())
    for result, leaf in zip(results, document.leaves()):
        for item in result:
            item.attach(leaf)
    return document


@pytest.fixture
def demo_page_segmented_twice(demo_image):
    document = Document(demo_image)
    for _ in range(2):
        results = dummy_segmentation_model(document.segments())
        for result, leaf in zip(results, document.leaves()):
            for item in result:
                item.attach(leaf)
    results = dummy_text_recognition_model(document.segments())
    for result, leaf in zip(results, document.leaves()):
        for item in result:
            item.attach(leaf)
    return document


@pytest.fixture
def demo_page_segmented_thrice(demo_image):
    document = Document(demo_image)
    for _ in range(3):
        results = dummy_segmentation_model(document.segments())
        for result, leaf in zip(results, document.leaves()):
            for item in result:
                item.attach(leaf)
    results = dummy_text_recognition_model(document.segments())
    for result, leaf in zip(results, document.leaves()):
        for item in result:
            item.attach(leaf)
    return document


@pytest.fixture
def demo_collection_unsegmented(demo_image):
    n_images = 5
    vol = Collection([demo_image] * n_images)
    return vol


@pytest.fixture
def demo_collection_segmented(demo_image):
    n_images = 5
    vol = Collection([demo_image] * n_images)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_collection_segmented_nested(demo_image):
    n_images = 5
    vol = Collection([demo_image] * n_images)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_collection_segmented_nested_with_text(demo_image):
    n_images = 5
    vol = Collection([demo_image] * n_images)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    result = dummy_text_recognition_model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_collection_with_text(demo_image):
    n_images = 1
    vol = Collection([demo_image] * n_images)
    result = dummy_text_recognition_model(vol.segments())
    vol.update(result)
    return vol


def dummy_segmentation(shape):
    width, height = shape
    h = height // 10
    regions = []
    for x in range(5):
        shape = Bbox(0, x * h, width, x * (h + 1)).polygon()
        regions.append(Region(shape))
    return regions


def dummy_text_recognition():
    n_candidates = 3
    texts = []
    for _ in range(n_candidates):
        words = ["".join(random.choices(ascii_letters, k=5)) for _ in range(5)]
        text = " ".join(words)
        texts.append(Text(text, random.random()))
    return texts


def dummy_segmentation_model(images) -> list[list[Region]]:
    return [dummy_segmentation(image.size) for image in images]


def dummy_text_recognition_model(images) -> list[list[Text]]:
    return [dummy_text_recognition() for _ in images]
