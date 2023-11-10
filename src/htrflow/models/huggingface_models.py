from transformers import pipeline

from htrflow.models.utils import check_device_to_use


class HuggingFaceModel:
    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: str = None, device: str = None):
        device = check_device_to_use(device)
        pipe = pipeline("image-to-text", model="microsoft/trocr-large-handwritten", device_map="auto")

        return pipe
