import random

import cv2
import lorem
import numpy as np
import pytest

from htrflow.results import Result
from htrflow.volume import volume


@pytest.fixture
def demo_image():
    return "examples/images/pages/A0068699_00021.jpg"


@pytest.fixture
def demo_page_unsegmented(demo_image):
    node = volume.PageNode(demo_image)
    result = dummy_text_recognition_model([node.image])
    node.add_data(**result[0].data)
    node.relabel()
    return node


@pytest.fixture
def demo_page_segmented_once(demo_image):
    node = volume.PageNode(demo_image)
    results = dummy_segmentation_model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        node.create_segments(result.segments)
    results = dummy_text_recognition_model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        leaf.add_data(**result.data)
    node.relabel()
    return node


@pytest.fixture
def demo_page_segmented_twice(demo_image):
    node = volume.PageNode(demo_image)
    for _ in range(2):
        results = dummy_segmentation_model(node.segments())
        for result, leaf in zip(results, node.leaves()):
            leaf.create_segments(result.segments)
    results = dummy_text_recognition_model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        leaf.add_data(**result.data)
    node.relabel()
    return node


@pytest.fixture
def demo_page_segmented_thrice(demo_image):
    node = volume.PageNode(demo_image)
    for _ in range(3):
        results = dummy_segmentation_model(node.segments())
        for result, leaf in zip(results, node.leaves()):
            leaf.create_segments(result.segments)
    results = dummy_text_recognition_model(node.segments())
    for result, leaf in zip(results, node.leaves()):
        leaf.add_data(**result.data)
    node.relabel()
    return node


@pytest.fixture
def demo_collection_unsegmented(demo_image):
    n_images = 5
    vol = volume.Collection([demo_image] * n_images)
    return vol


@pytest.fixture
def demo_collection_segmented(demo_image):
    n_images = 5
    vol = volume.Collection([demo_image] * n_images)
    result = dummy_segmentation_model(vol.images())
    vol.update(result)
    return vol


@pytest.fixture
def demo_collection_segmented_nested(demo_image):
    n_images = 5
    vol = volume.Collection([demo_image] * n_images)
    result = dummy_segmentation_model(vol.images())
    vol.update(result)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_collection_segmented_nested_with_text(demo_image):
    n_images = 5
    vol = volume.Collection([demo_image] * n_images)
    result = dummy_segmentation_model(vol.images())
    vol.update(result)
    result = dummy_segmentation_model(vol.segments())
    vol.update(result)
    result = dummy_text_recognition_model(vol.segments())
    vol.update(result)
    return vol


@pytest.fixture
def demo_collection_with_text(demo_image):
    n_images = 1
    vol = volume.Collection([demo_image] * n_images)
    result = dummy_text_recognition_model(vol.images())
    vol.update(result)
    return vol


def dummy_segmentation(shape):
    n_segments = 5
    scores = [random.random() for _ in range(n_segments)]
    labels = ["class_label"] * n_segments
    return Result.segmentation_result(
        shape,
        {},
        masks=[dummy_mask(shape) for _ in range(n_segments)],
        scores=scores,
        labels=labels,
    )


def dummy_text_recognition():
    n_candidates = 3
    return Result.text_recognition_result(
        {},
        texts=[lorem.sentence() for _ in range(n_candidates)],
        scores=[random.random() for _ in range(n_candidates)],
    )


def dummy_mask(shape):
    h, w = shape
    mask = np.zeros(shape, np.uint8)
    x, y = random.randrange(0, w), random.randrange(0, h)
    cv2.ellipse(mask, (x, y), (w // 8, h // 8), 0, 0, 360, color=(255,), thickness=-1)
    return mask


def dummy_segmentation_model(images):
    return [dummy_segmentation(image.shape[:2]) for image in images]


def dummy_text_recognition_model(images):
    return [dummy_text_recognition() for _ in images]
