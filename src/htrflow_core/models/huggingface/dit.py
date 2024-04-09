import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification


def dit_predict2(image):
    processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip", cache_dir=".cache")
    model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip", cache_dir=".cache")

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def dit_predict(image):
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "microsoft/dit-base-finetuned-rvlcdip", cache_dir=".cache"
    )
    model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip", cache_dir=".cache")

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    import requests
    from PIL import Image

    url = "https://github.com/Swedish-National-Archives-AI-lab/htrflow_core/blob/a1b4b31f9a8b7c658a26e0e665eb536a0d757c45/data/demo_image.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    results = dit_predict([image])

    print(results)
