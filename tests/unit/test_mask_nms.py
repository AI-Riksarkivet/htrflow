import random

import numpy as np
import pytest

# from htrflow.postprocess.mask_nms import multiclass_mask_nms, mask_nms
from htrflow.results import Result, Segment


@pytest.fixture
def results_with_mask():
    orig_shape = (200, 200)
    mask_a = np.zeros(orig_shape, dtype=np.uint8)
    mask_a[20:70, 20:70] = 1
    mask_b = np.zeros(orig_shape, dtype=np.uint8)
    mask_b[10:180, 10:180] = 1
    mask_c = np.zeros(orig_shape, dtype=np.uint8)
    mask_c[30:70, 30:70] = 1

    segment_a = Segment(mask=mask_a, class_label="class_1", orig_shape=orig_shape)
    segment_b = Segment(mask=mask_b, class_label="class_2", orig_shape=orig_shape)
    segment_c = Segment(mask=mask_c, class_label="class_1", orig_shape=orig_shape)

    return Result(metadata={}, segments=[segment_a, segment_b, segment_c])


def generate_random_masks(num_masks, image_size=(200, 200), num_classes=3):
    random.seed(42)
    np.random.seed(42)
    masks = []
    classes = [f"class_{i+1}" for i in range(num_classes)]
    for _ in range(num_masks):
        w, h = random.randint(20, 100), random.randint(20, 40)  # Random width and height
        x, y = random.randint(0, image_size[0] - w), random.randint(0, image_size[1] - h)  # Random position
        mask = np.zeros(image_size, dtype=np.uint8)
        mask[y : y + h, x : x + w] = 1
        class_label = random.choice(classes)
        masks.append(Segment(mask=mask, class_label=class_label, orig_shape=image_size))
    return masks


def simulate_large_dataset():
    num_masks = 100  # Simulating a large number of masks
    segments = generate_random_masks(num_masks)
    return Result(metadata={}, segments=segments)


# TODO: add expected_list... to assert with


@pytest.fixture
def large_dataset():
    # Assuming this function generates the dataset you're using to test
    return simulate_large_dataset(100)  # Adjust the number to match your dataset size


def test_multiclass_mask_nms(results_with_mask):
    # TODO: test work for different labels
    pass


def test_mask_nms(results_with_mask):
    # TODO
    pass
