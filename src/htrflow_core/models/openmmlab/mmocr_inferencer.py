from htrflow.structures.text_rec_result import TextRecResult

from htrflow_core.inferencers.base_inferencer import BaseInferencer
from htrflow_core.models.openmmlab_loader import OpenmmlabModel


class MMOCRInferencer(BaseInferencer):
    def __init__(self, text_rec_model: OpenmmlabModel):
        self.region_model = text_rec_model

    def preprocess():
        pass

    def predict(self, imgs):
        # image = mmcv.imread(input_image)

        result_raw = self.region_model(imgs, batch_size=8)
        batch_result = self.postprocess(result_raw)
        return batch_result

    def postprocess(self, result_raw):
        batch_result = [TextRecResult(text=x["text"], score=x["scores"]) for x in result_raw["predictions"]]

        return batch_result
