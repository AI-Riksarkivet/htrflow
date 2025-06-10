"""
This module holds the base data structures
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict
from itertools import chain
from typing import Generator, Iterable, Iterator, Sequence

import numpy as np

from htrflow import serialization
from htrflow.results import TEXT_RESULT_KEY, RecognizedText, Result, Segment
from htrflow.utils import imgproc
from htrflow.utils.geometry import Bbox, Point, Polygon
from htrflow.volume.node import Node


logger = logging.getLogger(__name__)


class ImageNode(Node, ABC):

    @property
    def coord(self) -> Point:
        """Coordinate of this node's top left corner, relative to original image"""
        return self.bbox.p1

    @property
    def width(self) -> int:
        """Width of this node"""
        return self.bbox.width

    @property
    def height(self) -> int:
        """Height of this node"""
        return self.bbox.height

    @property
    @abstractmethod
    def bbox(self) -> Bbox:
        """Bounding box of this node"""
        pass

    @property
    def polygon(self) -> Polygon:
        return self.bbox.polygon()

    def __str__(self) -> str:
        s = f"{self.height}x{self.width} node ({self.label}) at ({self.coord.x}, {self.coord.y})"
        if self.text:
            s += f": {self.text}"
        return s

    def clear_images(self):
        """Remove cached images"""
        for node in self.traverse():
            del node._image
            node._image = None

    @property
    def image(self):
        """The image this node represents"""
        if self._image is None:
            self._image = self._load_image()
        return self._image

    @abstractmethod
    def _load_image(self):
        pass

    @property
    def text(self) -> str | None:
        """Text of this region, if available"""
        if text_result := self.get(TEXT_RESULT_KEY):
            return text_result.top_candidate()
        return None

    @property
    def text_result(self) -> RecognizedText | None:
        return self.get(TEXT_RESULT_KEY, None)

    def is_word(self):
        return self.text is not None and self.parent and self.parent.is_line()

    def is_line(self):
        return self.text is not None and self.parent and self.parent.text is None

    def update(self, result: Result):
        """Update node with result"""
        if result.segments:
            self.create_segments(result.segments)
        self.add_data(**result.data)

    def create_segments(self, segments: Sequence[Segment]) -> None:
        """Segment this node"""
        self.children = [SegmentNode(segment, self) for segment in segments]
        del self._image
        self._image = None

    def has_regions(self) -> bool:
        return all(child.text is None for child in self.children)

    def segments(self) -> "ImageGenerator":
        return ImageGenerator(self.leaves())

    def is_region(self) -> bool:
        return bool(self.children) and not self.text


class SegmentNode(ImageNode):
    """A node representing a segment of a page"""

    def __init__(self, segment: Segment, parent: ImageNode):
        segment.move(parent.coord)
        label = segment.class_label or "region"
        super().__init__(parent=parent, label=label)
        self.add_data(**segment.data)
        self._segment = segment
        self._image = None

    @property
    def bbox(self) -> Bbox:
        return self._segment.bbox

    @property
    def polygon(self) -> Polygon:
        return self._segment.polygon

    def _load_image(self):
        img = imgproc.crop(self.parent.image, self.bbox.move(-self.parent.coord))
        mask = self._segment.mask
        if mask is not None:
            img = imgproc.mask(img, mask)
        return img

    def asdict(self) -> dict:
        return super().asdict() | {
            "segmentation_label": self._segment.class_label,
            "segmentation_confidence": self._segment.score,
            "bbox": asdict(self.bbox),
            "polygon": str(self.polygon),
        }


class PageNode(ImageNode):
    """A node representing a page / input image"""

    def __init__(self, image_path: str):
        self.path = image_path
        self._image = self._load_image()
        label = os.path.splitext(os.path.basename(image_path))[0]

        super().__init__(parent=None, label=label)
        self.add_data(
            file_name=os.path.basename(image_path),
            image_path=image_path,
            image_name=label,
        )

    @property
    def bbox(self) -> Bbox:
        height, width = self.image.shape[:2]
        return Bbox(0, 0, width, height)

    def _load_image(self):
        return imgproc.read(self.path)

    def asdict(self) -> dict:
        return super().asdict() | {
            "height": self.height,
            "width": self.width,
        }


class Collection:
    pages: list[PageNode]
    _DEFAULT_LABEL = "untitled_collection"

    def __init__(self, paths: Sequence[str], label: str | None = None):
        """Initialize collection

        Arguments:
            paths: A list of paths to images
            label: An optional label describing the collection. If not given,
                the label will be set to the input paths' first shared
                parent directory, and if no such directory exists, it will
                default to "untitled_collection".
        """
        self.pages = paths2pages(paths)
        self.label = label or _common_basename(paths) or Collection._DEFAULT_LABEL
        logger.info("Initialized collection '%s' with %d pages", label, len(self.pages))

    def __iter__(self) -> Iterator[PageNode]:
        return iter(self.pages)

    def __getitem__(self, idx) -> ImageNode:
        if isinstance(idx, tuple):
            i, *rest = idx
            return self.pages[i][rest]
        return self.pages[idx]

    def traverse(self, filter):
        return chain(*[page.traverse(filter) for page in self])

    @classmethod
    def from_directory(cls, path: str) -> "Collection":
        """Initialize a collection from a directory

        Sets the collection label to the directory name.

        Arguments:
            path: A path to a directory of images.
        """
        paths = [os.path.join(path, file) for file in sorted(os.listdir(path))]
        return cls(paths)

    @classmethod
    def from_pickle(cls, path: str) -> "Collection":
        """Initialize a collection from a pickle file

        Arguments:
            path: A path to a previously pickled collection instance
        """
        with open(path, "rb") as f:
            collection = pickle.load(f)

        if not isinstance(collection, Collection):
            raise pickle.UnpicklingError(f"Unpickling {path} did not return a Collection instance.")

        logger.info("Loaded collection '%s' from %s", collection.label, path)
        return collection

    def __str__(self):
        return f"collection label: {self.label}\ncollection tree:\n" + "\n".join(child.tree2str() for child in self)

    def images(self) -> "ImageGenerator":
        """Yields the collection's original input images"""
        return ImageGenerator(page for page in self.pages)

    def segments(self) -> "ImageGenerator":
        """Yield the active segments' images"""
        return ImageGenerator(self.active_leaves())

    def leaves(self) -> Iterator[ImageNode]:
        yield from chain(*[page.leaves() for page in self])

    def active_leaves(self) -> Generator[ImageNode, None, None]:
        """Yield the collection's active leaves

        Here, an "active leaf" is a leaf node whose depth is equal to
        the maximum depth of the tree. In practice, this means that the
        node was segmented in the previous step (or is a fresh PageNode).
        Inactive leaves are leaves that weren't segmented in the
        previous step, and thus are higher up in the tree than the
        other leaves. These should typically not updated in the next
        steps.
        """
        if self.pages:
            max_depth = max(page.max_depth() for page in self)
            for leaf in self.leaves():
                if leaf.depth == max_depth:
                    yield leaf

    def update(self, results: list[Result]) -> None:
        """Update the collection with model results

        Arguments:
            results: A list of results where the i:th result
                corresponds to the collection's i:th active leaf node.
        """
        leaves = list(self.active_leaves())
        if len(leaves) != len(results):
            raise ValueError(f"Size of input ({len(results)}) does not match the size of the tree ({len(leaves)})")

        for leaf, result in zip(leaves, results):
            leaf.update(result)
        self.relabel()

    def save(
        self,
        directory: str = "outputs",
        serializer: str | serialization.Serializer = "alto",
    ) -> None:
        """Save collection

        Arguments:
            directory: Output directory
            serializer: What serializer to use, either a string name (e.g.,
                "alto") or a Serializer instance. See serialization.supported_formats()
                for available string options.
        """
        serialization.save_collection(self, serializer, directory)

    def relabel(self):
        for page in self:
            page.relabel()


class ImageGenerator:
    """A generator with __len__

    Wrapper around `nodes` that provides a generator over the nodes'
    images and implements len(). This way, there is no need to load
    all images into memory at once, but the length of the generator
    is known beforehand (which is typically not the case), which is
    handy in some cases, e.g., when using tqdm progress bars.
    """

    def __init__(self, nodes: Iterable[ImageNode]):
        self._nodes = list(nodes)

    def __iter__(self) -> Iterator[np.ndarray]:
        for _node in self._nodes:
            yield _node.image

    def __len__(self) -> int:
        return len(self._nodes)


def paths2pages(paths: Sequence[str]) -> list[PageNode]:
    """Create PageNodes

    Creates PageNodes from the given paths. Any path pointing to a file
    that cannot be read or interpreted as an image will be ignored.

    Arguments:
        paths: A sequence of paths pointing to image files.

    Returns:
        A list of PageNodes corresponding to the input paths.
    """
    pages = []
    for path in sorted(paths):
        try:
            page = PageNode(path)
        except imgproc.ImageImportError as e:
            logger.warning(e)
            continue
        pages.append(page)
    return pages


def _common_basename(paths: Sequence[str]):
    """Given a sequence of paths, returns the name of their first shared parent directory"""
    if len(paths) > 1:
        return os.path.basename(os.path.commonpath(paths))
    return os.path.basename(os.path.dirname(paths[0]))
