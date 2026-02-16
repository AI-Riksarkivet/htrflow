from htrflow.document import Region, Text
from htrflow.utils.geometry import Bbox


def _simple_word_segmentation(region: Region):
    text = region.transcription[0].text
    words = text.split()
    pixels_per_char = region.polygon.width // len(text)
    x1, x2 = 0, 0

    regions = []
    for word in words:
        x2 = min(x1 + pixels_per_char * (len(word) + 1), region.polygon.width)
        polygon = Bbox(x1, 0, x2, region.polygon.height).polygon()
        region = Region(polygon=polygon, transcription=[Text(word)])
        regions.append(region)
        x1 = x2
    return regions


def simple_word_segmentation(regions: list[Region]):
    return [_simple_word_segmentation(region) for region in regions]
