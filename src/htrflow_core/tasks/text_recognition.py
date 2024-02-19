
from htrflow_core.inferencers.base_inferencer import BaseInferencer
from htrflow_core.tasks.base_task import BaseTask


class TextRecognition(BaseTask):
    def __init__(self, text_rec_inferencer: BaseInferencer):
        self.text_rec_inferencer= text_rec_inferencer

    def preprocess():
        pass

    def run(self, input_images):
        self.text_rec_inferencer.predict(input_images)
        print(input_images)

    def postprocess():
        pass


if __name__ == "__main__":
    from htrflow_core.inferencers.openmmlab.mmocr_inferencer import MMOCRInferencer
    from htrflow_core.models.openmmlab_loader import OpenmmlabModelLoader

    satrn_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/satrn_htr", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    # lines_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_lines", cache_dir="./models")

    text_recognizer = TextRecognition(MMOCRInferencer(satrn_model))


    print(text_recognizer.text_rect_inferencer)
