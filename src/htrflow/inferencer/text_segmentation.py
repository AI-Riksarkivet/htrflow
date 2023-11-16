from htrflow.inferencer.base_inferencer import BaseInferencer
from htrflow.models.openmmlab_models import OpenmmlabModel


class TextSegmentation:
    def __init__(self, segmentation_model: BaseInferencer):
        self.segmentation_model = segmentation_model

    def predict(self, input_images):
        self.segmentation_model.predict(input_images)
        print(input_images)

        # Should return to the dataframe


class TextSegInstanceChecker:
    @staticmethod
    def create_openmmlab_model(inferencer_scope):
        inferencers = {
            # OpenmmlabsFramework.MMDET.value: MMDetInferencer,
        }

        if inferencer_scope in inferencers:
            return


if __name__ == "__main__":
    region_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    # lines_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_lines", cache_dir="./models")
    # text_rec_model = OpenmmlabModel.from_pretrained("Riksarkivet/satrn_htr", cache_dir="./models")

    print(region_model)
