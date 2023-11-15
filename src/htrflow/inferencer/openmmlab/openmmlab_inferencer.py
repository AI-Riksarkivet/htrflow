import mmcv

from htrflow.inferencer.base_inferencer import BaseInferencer


class OpenmmlabInferencer(BaseInferencer):
    def __init__(self, region_model):
        self.region_model = region_model

    def preprocess():
        pass

    def predict(self, input_image):
        image = mmcv.imread(input_image)

        # prediction logic using HuggingFace
        return image

    def postprocess():
        pass
