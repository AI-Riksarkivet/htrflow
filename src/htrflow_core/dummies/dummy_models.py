import random
from typing import List, Literal, Optional

import cv2
import lorem  # type: ignore
import numpy as np

from htrflow_core.models.base_model import BaseModel
from htrflow_core.results import RecognizedText, Result, Segment
from htrflow_core.utils import imgproc
from htrflow_core.utils.geometry import bbox2mask


"""
This module contains dummy models
"""


class SegmentationModel(BaseModel):
    def __init__(self, segment_type: Literal["mask", "bbox"] = "mask") -> None:
        super().__init__()
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
                        Segment(
                            orig_shape=(h, w), mask=mask, score=score, class_label=label if label else randomlabel()
                        )
                    )
                else:
                    bbox = randombox(h, w)
                    segments.append(
                        Segment(
                            orig_shape=(h, w), bbox=bbox, score=score, class_label=label if label else randomlabel()
                        )
                    )

            results.append(Result(metadata, segments))
        return results


class RecognitionModel(BaseModel):
    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        metadata = generate_metadata(self)
        n = 2
        return [
            Result.text_recognition_result(
                metadata,
                RecognizedText(texts=[lorem.sentence() for _ in range(n)], scores=[random.random() for _ in range(n)]),
            )
            for _ in images
        ]


class ClassificationModel(BaseModel):
    """Model that classifies input images as potato dishes"""

    def _predict(self, images: list[np.ndarray]) -> list[Result]:
        classes = ["baked potato", "french fry", "raggmunk"]
        return [
            Result(metadata={"model": "Potato classifier 2000"}, data=[{"classification": random.choice(classes)}])
            for _ in images
        ]


def generate_metadata(model: SegmentationModel | RecognitionModel) -> dict:
    """Generates metadata for a given model."""
    model_name = model.__class__.__name__
    return {"model_name": model_name}


def randombox(h: int, w: int) -> List[int]:
    """Makes a box that is a height/3-by-width/4 and places it at a random location"""
    x = random.randrange(0, 4 * w // 5)
    y = random.randrange(0, 5 * h // 6)
    return [x, y, x + w // 5, y + h // 6]


def randomlabel() -> str:
    return "region"


def randommask(h: int, w: int) -> np.ndarray:
    """Makes an elliptical mask and places it at a random location"""
    mask = np.zeros((h, w), np.uint8)
    x, y = random.randrange(0, w), random.randrange(0, h)
    cv2.ellipse(mask, (x, y), (w // 8, h // 8), 0, 0, 360, color=(255,), thickness=-1)
    return mask


def simple_word_segmentation(nodes) -> list[Result]:
    return [_simple_word_segmentation(node.image, node.text, node.get("segment")) for node in nodes]


def _simple_word_segmentation(image, text, segment=None):
    height, width = image.shape[:2]
    pixels_per_char = width // len(text)
    bboxes = []
    x1, x2 = 0, 0
    words = text.split()
    for word in words:
        x2 = min(x1 + pixels_per_char * (len(word) + 1), width)
        bboxes.append((x1, 0, x2, height))
        x1 = x2

    if segment and segment.mask is not None:
        masks = [imgproc.mask(segment.mask, bbox2mask(bbox, segment.mask.shape), fill=0) for bbox in bboxes]
        segments = [Segment(mask=mask, class_label="word") for mask in masks]
    else:
        segments = [Segment(bbox=bbox, class_label="word") for bbox in bboxes]
    texts = [RecognizedText([word], [0]) for word in words]
    r = Result(image, {"model": "simple word segmentation"}, segments, [{"text_result": text} for text in texts])
    return r
