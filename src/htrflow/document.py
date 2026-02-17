"""
HTRflow document model.
"""

import os
from abc import ABC, abstractmethod

from PIL import Image

from htrflow.utils.geometry import Bbox, Polygon
from htrflow.utils.imgproc import polygon_mask


class _RegionAttachment(ABC):
    """
    An ABC for anything that is attachable to Region instances.
    """

    @abstractmethod
    def attach(self, region: "Region"):
        """
        Attach self to the given region.
        """
        pass


class Text(_RegionAttachment):
    """
    A transcription.
    """

    text: str
    confidence: float | None

    def __init__(self, text: str, confidence: float | None = None):
        self.text = text
        self.confidence = confidence

    def attach(self, region: "Region"):
        region.transcription.append(self)


class Annotation(_RegionAttachment):
    def __init__(self, **annotations):
        self.annotations = annotations

    def attach(self, region: "Region"):
        region.annotations |= self.annotations


class Region(_RegionAttachment):
    annotations: dict
    regions: list["Region"]
    polygon: Polygon
    transcription: list[Text]

    def __init__(
        self,
        polygon: Polygon,
        regions: list["Region"] | None = None,
        transcription: list[Text] | None = None,
        **annotations,
    ):
        self.polygon = polygon
        self.regions = regions or []
        self.transcription = transcription or []
        self.annotations = annotations

    def attach(self, region: "Region"):
        region.regions.append(self)

    def traverse(self):
        return [self] + [node for child in self.regions for node in child.traverse()]

    def leaves(self):
        return [node for node in self.traverse() if node.is_leaf()]

    def is_leaf(self) -> bool:
        return not self.regions


class Document(Region):
    def __init__(self, image_path):
        self._image_path = image_path
        self.image_name, _ = os.path.splitext(os.path.basename(image_path))
        polygon = Bbox(0, 0, self.image.width, self.image.height).polygon()
        super().__init__(polygon)

    @property
    def image(self):
        return Image.open(self._image_path).convert("RGB")

    def segments(self):
        return ImageLoader(self)

    def update(self, nodes):
        leaves = [leaf for p in self.pages for leaf in p.leaves()]
        for leaf, nodes in zip(leaves, nodes):
            for node in nodes:
                node.attach(leaf)


class Collection:
    def __init__(self, paths):
        self.pages = [Document(path) for path in paths]

    def __iter__(self):
        return iter(self.pages)

    def segments(self):
        for page in self.pages:
            yield from ImageLoader(page)

    def update(self, nodes):
        leaves = [leaf for p in self.pages for leaf in p.leaves()]
        for leaf, nodes in zip(leaves, nodes):
            for node in nodes:
                node.attach(leaf)


class ImageLoader:
    def __init__(self, page):
        self.page = page

    def __iter__(self):
        yield from self._image_loader(self.page)

    def _image_loader(self, node, base_image=None):
        image = node.image if base_image is None else polygon_mask(base_image, node.polygon)
        if node.regions:
            for region in node.regions:
                yield from self._image_loader(region, image)
        else:
            yield image


def traverse(region: Region):
    return [region] + [subregion for region in region.regions for subregion in region.traverse()]
