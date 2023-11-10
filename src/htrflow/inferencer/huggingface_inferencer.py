from htrflow.inferencer.base_inferencer import BaseInferencer


class HuggingFaceInferencer(BaseInferencer):
    def __init__(self, region_model):
        self.region_model = region_model

    def preprocess():
        pass

    def predict(self, input_image):
        # prediction logic using HuggingFace
        return "some_result_hf"

    def postprocess():
        pass
