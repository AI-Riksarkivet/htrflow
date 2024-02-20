import pytest
from htrflow_core.inferencers.huggingface.trocr_inferencer import TrOCRInferencer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@pytest.fixture
def inferencer():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return TrOCRInferencer(model, processor)


@pytest.fixture
def images():
    return ["data/raw/trocr_demo_image.png", "data/raw/demo_image.jpg"]


def test_num_beams(inferencer, images):
    """Ensure that `num_beams=n` returns `n` texts"""
    num_beams = 2
    texts, _ = inferencer.predict(images, num_beams=num_beams)
    print(texts[0])
    assert len(texts[0]) == num_beams


def test_num_return_sequences(inferencer, images):
    """Ensure that `num_return_sequences=n` returns `n` texts"""
    num_return_sequences = 2
    texts, _ = inferencer.predict(images, num_return_sequences=num_return_sequences)
    assert len(texts[0]) == num_return_sequences