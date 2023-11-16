from htrflow.inferencer.huggingface.trocr_inferencer import TrocrInferencer
from htrflow.inferencer.openmmlab.mmocr_inferencer import MMDetInferencer
from htrflow.models.base_model import BaseModel
from htrflow.models.openmmlab_models import OpenmmlabModelLoader


class TextRecognition:
    def __init__(self, text_rec_model:  BaseModel):
        self.text_rect_inferencer = TextRecInstanceChecker(text_rec_model.framework)

    def predict(self, input_images):
        self.text_rec_model.predict(input_images)
        print(input_images)


class TextRecInstanceChecker:
    @staticmethod
    def get_inferencer(inferencer_scope):
        inferencers = {
            "mmocr": MMDetInferencer,
            "trocr": TrocrInferencer,
        }

        if inferencer_scope in inferencers:
            return inferencers[inferencer_scope]


if __name__ == "__main__":
    rtmdet_region_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    # lines_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_lines", cache_dir="./models")
    region_inferencer= TextRecognition(rtmdet_region_model)
    print(region_inferencer.text_rect_inferencer)
