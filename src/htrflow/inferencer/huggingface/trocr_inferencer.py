from htrflow.inferencer.base_inferencer import BaseInferencer


class TrocrInferencer(BaseInferencer):
    def __init__(self, region_model):
        self.region_model = region_model

    def preprocess():
        pass
        #processor

    def predict(self, input_image):
        # prediction logic using HuggingFace
        return "some_result_hf"
        #model.generate()

    def postprocess():
        pass
        #decoder
