from htrflow.inferencer.base_inferencer import BaseInferencer


class TextRecognition:
    def __init__(self, text_rec_model: BaseInferencer):
        self.text_rec_model = text_rec_model

    def predict(self, input_images):
        self.text_rec_model.predict(input_images)
        print(input_images)

        # Should return to the dataframe
