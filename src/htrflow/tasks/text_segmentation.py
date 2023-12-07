
from htrflow.inferencers.base_inferencer import BaseInferencer
from htrflow.tasks.base_task import BaseTask


class TextSegmentation(BaseTask):
    def __init__(self, text_seg_inferencer: BaseInferencer):
        self.text_rec_inferencer= text_seg_inferencer

    def preprocess(self):
        pass

    def run(self, dataset):
        input_images = dataset["pointer_to_image"]
        standard_dict = self.text_rec_inferencer.predict(input_images)

        process_standard_dict = self.postprocess(standard_dict)

        dataset["predictions_images"] = process_standard_dict

        print(input_images)

    def postprocess(self):
        pass

if __name__ == "__main__":

    from htrflow.inferencers.openmmlab.mmdet_inferencer import MMDetInferencer
    from htrflow.models.openmmlab_loader import OpenmmlabModelLoader

    rtmdet_region_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    text_segmenter= TextSegmentation(MMDetInferencer(rtmdet_region_model))

    print(text_segmenter.segmentation_inferencer)
