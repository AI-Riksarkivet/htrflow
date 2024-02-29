import pytest
from htrflow.models.openmmlab_loader import OpenmmlabModelLoader
from mmdet.apis import DetInferencer
from mmocr.apis import TextRecInferencer


@pytest.mark.parametrize(
    "model_id, expected_type",
    [
        ("Riksarkivet/satrn_htr", TextRecInferencer),
        ("Riksarkivet/rtmdet_regions", DetInferencer),
        ("invalid_model_id", None),
    ],
)
def test_from_pretrained__returns_correct_type(model_id, expected_type):
    result = OpenmmlabModelLoader.from_pretrained(model_id)

    if expected_type is None:
        assert result is None
    else:
        assert isinstance(result, expected_type)
