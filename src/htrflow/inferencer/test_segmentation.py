from htrflow.inferencer.base_inferencer import BaseInferencer


class RegionInferencer:
    def __init__(self, region_model: BaseInferencer):
        self.region_model = region_model

    def predict(self, input_images, batches_size):
        self.region_model.predict(input_images)
        print(input_images)
