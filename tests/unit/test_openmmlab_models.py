import pytest
from mmdet.apis import DetInferencer
from mmocr.apis import TextRecInferencer

from htrflow.models.openmmlab_models import OpenmmlabModel


@pytest.mark.parametrize(
    "model_id, expected_type",
    [
        ("Riksarkivet/satrn_htr", TextRecInferencer),
        ("Riksarkivet/rtmdet_regions", DetInferencer),
        ("invalid_model_id", None),
    ],
)
def test_from_pretrained__returns_correct_type(model_id, expected_type):
    result = OpenmmlabModel.from_pretrained(model_id)

    if expected_type is None:
        assert result is None
    else:
        assert isinstance(result, expected_type)
