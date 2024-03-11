import random
from typing import List, Literal, Optional

import cv2
import lorem  # type: ignore
import numpy as np

from htrflow_core.models.base_model import BaseModel
from htrflow_core.results import RecognizedText, Result, Segment


"""
This module contains dummy models
"""


class SegmentationModel(BaseModel):
    def __init__(self, segment_type: Literal["mask", "bbox"] = "mask") -> None:
        self.segment_type = segment_type

    def _predict(self, images: list[np.ndarray], label: Optional[str] = None) -> list[Result]:
        metadata = generate_metadata(self)

        results = []
        for image in images:
            n_segments = random.randint(1, 5)
            h, w, _ = image.shape
            segments = []
            for _ in range(n_segments):
                score = random.random()

                if self.segment_type == "mask":
                    mask = randommask(h, w)
                    segments.append(
                        Segment.from_mask(mask, score=score, class_label=label if label else randomlabel())
                    )
                else:
                    bbox = randombox(h, w)
                    segments.append(
                        Segment.from_bbox(bbox, score=score, class_label=label if label else randomlabel())
                    )

            results.append(Result(image, metadata, segments, []))
        return results


class RecognitionModel(BaseModel):
    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        metadata = generate_metadata(self)
        n = 2
        return [
            Result.text_recognition_result(
                image,
                metadata,
                RecognizedText(texts=[lorem.sentence() for _ in range(n)], scores=[random.random() for _ in range(n)]),
            )
            for image in images
        ]


def generate_metadata(model: SegmentationModel | RecognitionModel) -> dict:
    """Generates metadata for a given model."""
    model_name = model.__class__.__name__
    return {"model_name": model_name}


def randombox(h: int, w: int) -> List[int]:
    """Makes a box that is a height/3-by-width/4 and places it at a random location"""
    x = random.randrange(0, 4 * w // 5)
    y = random.randrange(0, 5 * h // 6)
    return [x, x + w // 5, y, y + h // 6]


def randomlabel() -> str:
    return "region"


def randommask(h: int, w: int) -> np.ndarray:
    """Makes an elliptical mask and places it at a random location"""
    mask = np.zeros((h, w), np.uint8)
    x, y = random.randrange(0, w), random.randrange(0, h)
    cv2.ellipse(mask, (x, y), (w // 8, h // 8), 0, 0, 360, color=(255,), thickness=-1)
    return mask


def simple_word_segmentation(nodes) -> list[Result]:
    return [_simple_word_segmentation(node.image, node.text) for node in nodes]


def _simple_word_segmentation(image, text):
    if random.random() < 0:
        return Result(image, {})

    height, width = image.shape[:2]
    pixels_per_char = width // len(text)
    bboxes = []
    x1, x2 = 0, 0
    words = text.split()
    for word in words:
        x2 = min(x1 + pixels_per_char * len(word), width)
        bboxes.append((x1, x2, 0, height))
        x1 = x2 + pixels_per_char  # add a "whitespace"

    segments = [Segment.from_bbox(bbox, class_label="word") for bbox in bboxes]
    texts = [RecognizedText([word], [0]) for word in words]
    r = Result(image, {"model": "simple word segmentation"}, segments, texts)
    return r
