from htrflow.results import Result
from htrflow.utils.geometry import bbox2mask
from htrflow.utils.imgproc import mask
from htrflow.volume.volume import SegmentNode


def _simple_word_segmentation(node: SegmentNode):
    text = node.text
    words = text.split()
    pixels_per_char = node.width // len(text)
    x1, x2 = 0, 0
    bboxes = []
    for word in words:
        x2 = min(x1 + pixels_per_char * (len(word) + 1), node.width)
        bboxes.append((x1, 0, x2, node.height))
        x1 = x2
    masks = [mask(node.mask, bbox2mask(bbox, node.mask.shape), fill=0) for bbox in bboxes]

    return Result.word_segmentation_result(
        orig_shape=(node.height, node.width),
        metadata={},
        masks=masks,
        words=words,
    )


def simple_word_segmentation(nodes: list[SegmentNode]):
    return [_simple_word_segmentation(node) for node in nodes]
