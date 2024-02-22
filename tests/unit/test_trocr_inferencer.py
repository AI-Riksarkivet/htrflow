import pytest

from htrflow_core.inferencers.huggingface.trocr_inferencer import TrOCR


@pytest.fixture
def model():
    return TrOCR()


@pytest.fixture
def images():
    return ["data/demo_image.jpg"] * 2


def test_num_beams(model, images):
    """Ensure that `num_beams=n` returns `n` texts"""
    num_beams = 2
    results = model(images, num_beams=num_beams)
    assert len(results[0].texts) == num_beams


def test_num_return_sequences(model, images):
    """Ensure that `num_return_sequences=n` returns `n` texts"""
    num_return_sequences = 2
    results = model(images, num_return_sequences=num_return_sequences)
    assert len(results[0].texts) == num_return_sequences
