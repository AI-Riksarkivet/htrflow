import numpy as np
import pytest


pytest.importorskip("laia", reason="Teklia dependencies not installed")

from htrflow.models.teklia.pylaia import PyLaia
from htrflow.results import Result
from htrflow.utils.imgproc import read


NORMAL_IMAGE_PATH = "examples/images/lines/A0068699_00021_region0_line1.jpg"
NOISE_IMAGE_SHAPE = (800, 128, 3)

def create_noise_image(dimensions: tuple[int, int, int]) -> np.ndarray:
    width, height, channels = dimensions
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

@pytest.fixture(scope="module")
def pylaia_model() -> PyLaia:
    return PyLaia("Teklia/pylaia-belfort")

@pytest.mark.teklia
def test_recognition_on_normal_image(pylaia_model):

    image = read(NORMAL_IMAGE_PATH)

    results = pylaia_model([image])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, Result)

    recognized_text = result.texts[0]
    assert isinstance(recognized_text, str)
    assert recognized_text.strip()

@pytest.mark.teklia
def test_handling_of_noise_image(pylaia_model):
    noise_image = create_noise_image(NOISE_IMAGE_SHAPE)
    results = pylaia_model([noise_image])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, Result)

    assert result.texts[0] == ""
    assert result.confidences[0] == 0.0
