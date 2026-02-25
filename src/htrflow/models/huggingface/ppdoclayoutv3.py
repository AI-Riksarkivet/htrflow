from transformers import AutoImageProcessor, AutoModelForObjectDetection

from htrflow.document import Region
from htrflow.models.base_model import BaseModel
from htrflow.utils.geometry import Polygon


class PPDocLayoutV3(BaseModel):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModelForObjectDetection.from_pretrained(model)
        self.model.to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model)

    def _predict(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = [image.size[::-1] for image in images]

        results = []
        for result in self.processor.post_process_object_detection(outputs, target_sizes=target_sizes):
            regions = []
            for score, label, polygon in zip(result["scores"], result["labels"], result["polygon_points"]):
                polygon = Polygon(polygon)
                label = self.model.config.id2label[int(label)]
                region = Region(polygon, score=score, label=label)
                regions.append(region)
            results.append(regions)
        return results
