from htrflow_core.models.base_model import BaseModel
from htrflow_core.results import RecognitionResult


class MMOCRInferencer(BaseModel):
    def __init__(self, model, *args):
        self.region_model = text_rec_model

    def predict(self, imgs):
        # image = mmcv.imread(input_image)

        result_raw = self.region_model(imgs, batch_size=8)
        batch_result = self.postprocess(result_raw)
        return batch_result

    def postprocess(self, result_raw):
        batch_result = [RecognitionResult(text=x["text"], score=x["scores"]) for x in result_raw["predictions"]]

        return batch_result
