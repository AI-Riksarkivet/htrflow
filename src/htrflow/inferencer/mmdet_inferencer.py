import numpy as np

from htrflow.inferencer.base_inferencer import BaseInferencer
from htrflow.structures.result import Result, SegResult
from htrflow.utils.helper import timing_decorator


class MMDetInferencer(BaseInferencer):
    def __init__(self, region_model, parent_result: Result = None):
        self.region_model = region_model
        self.parent_result = parent_result

    def preprocess():
        pass

    @timing_decorator
    def predict(self, input_images, batch_size: int = 1, pred_score_thr: float = 0.3):
        # image = mmcv.imread(input_images)

        result_raw = self.region_model(input_images, batch_size, return_datasample=True)
        batch_result = self.postprocess(result_raw, input_images)

        return batch_result

    @timing_decorator
    def postprocess(self, result_raw, input_images):
        # hm jag behöver nog inte nästla lines i seg_result, eller kanske ett bra sätt för att skilja på lines within regions och lines within page?
        # i postprocess... eller hmm, jag vill inte lägga för mycket här, jag tror att jag lägger det i en egen klass istället...

        batch_result = [
            Result(
                img_shape=np.shape(y)[0:2],
                segmentation=SegResult(
                    labels=x.pred_instances.labels.clone(),
                    bboxes=x.pred_instances.bboxes.clone(),
                    masks=x.pred_instances.masks.clone(),
                    scores=x.pred_instances.scores.clone(),
                ),
            )
            for x, y in zip(result_raw["predictions"], input_images)
        ]

        return batch_result
