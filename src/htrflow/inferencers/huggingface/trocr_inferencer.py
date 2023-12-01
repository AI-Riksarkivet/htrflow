
import torch
from transformers import GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel

from htrflow.helper.timing_decorator import timing_decorator
from htrflow.inferencers.base_inferencer import BaseInferencer


class TrocrInferencer(BaseInferencer):
    def __init__(self, model : VisionEncoderDecoderModel , processor: TrOCRProcessor, gen_config: GenerationConfig = None, skip_special_tokens=True ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.processor = processor
        self.gen_config = gen_config #https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores
        self.skip_special_tokens= skip_special_tokens

    def preprocess(self, images):
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values #torch.FloatTensor of shape (batch_size, num_channels, height, width)
        return pixel_values.to(self.device)

    @timing_decorator
    def predict(self, images):

        pixel_values = self.preprocess(images)

        return_dict = self.model.generate(inputs=pixel_values, num_beams=1, generation_config=self.gen_config, output_scores=True, return_dict_in_generate=True)

        generated_ids, scores = return_dict['sequences'], return_dict['scores']
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=self.skip_special_tokens)

        return generated_texts, scores


    def postprocess(self):
        pass
        #decoder


        #        labels = self.processor.tokenizer(text,
        #                                   padding="max_length",
        #                                   max_length=self.max_target_length).input_ids
        # # important: make sure that PAD tokens are ignored by the loss function
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        # encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # return encoding


if __name__ == "__main__":

    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # load image from the IAM database
    image1 = Image.open("/home/gabriel/Desktop/htrflow_core/data/raw/trocr_demo_image.png").convert("RGB")
    image2 = Image.open("/home/gabriel/Desktop/htrflow_core/data/raw/demo_image.jpg").convert("RGB")

    images = image1  #[image1,image2]

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

    trocr_rec = TrocrInferencer(model, processor)


    generated_texts, scores= trocr_rec.predict(images)

    print(generated_texts)
    # print(scores)




