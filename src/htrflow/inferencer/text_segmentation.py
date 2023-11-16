from htrflow.inferencer.openmmlab.mmdet_inferencer import MMDetInferencer
from htrflow.models.base_model import BaseModel
from htrflow.models.openmmlab_models import OpenmmlabModelLoader


class TextSegmentation:
    def __init__(self, segmentation_model: BaseModel):
        self.segmentation_inferencer= TextSegInstanceChecker.get_inferencer(segmentation_model.framework)

    def predict(self, input_images):
        # self.segmentation_model.predict(input_images)
        print(input_images)

        # Should return to the dataframe


class TextSegInstanceChecker:
    @staticmethod
    def get_inferencer(inferencer_scope):
        inferencers = {
            "mmdet": MMDetInferencer,
        }

        if inferencer_scope in inferencers:
            return inferencers[inferencer_scope]


if __name__ == "__main__":
    rtmdet_region_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    # lines_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_lines", cache_dir="./models")
    region_inferencer= TextSegmentation(rtmdet_region_model)
    print(region_inferencer.segmentation_inferencer)
