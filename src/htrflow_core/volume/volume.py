"""
This module holds the base data structures
"""
import logging
import os
import pickle
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import chain
from typing import Generator, Iterable, Iterator, Optional, Sequence

import numpy as np

from htrflow_core import serialization
from htrflow_core.results import Result, Segment
from htrflow_core.utils import imgproc
from htrflow_core.utils.geometry import Bbox, Mask, Point, Polygon
from htrflow_core.volume import node


logger = logging.getLogger(__name__)


class ImageNode(node.Node, ABC):
    parent: "ImageNode | None"
    children: list["ImageNode"]

    def __init__(
        self,
        height: int,
        width: int,
        coord: Point = Point(0, 0),
        polygon: Polygon | None = None,
        mask: Mask | None = None,
        parent: "ImageNode | None" = None,
        label: str | None = None,
    ):
        super().__init__(parent=parent, label=label)
        self.height = height
        self.width = width
        self.coord = coord
        self.bbox = Bbox(0, 0, width, height).move(coord)
        self.polygon = polygon or self.bbox.polygon()
        self.mask = mask

    def __str__(self) -> str:
        s = f"{self.height}x{self.width} node ({self.label}) at ({self.coord.x}, {self.coord.y})"
        if self.text:
            s += f": {self.text}"
        return s

    @property
    @abstractmethod
    def image(self) -> np.ndarray:
        """Image of the region this node represents"""

    @property
    def text(self) -> str | None:
        """Text of this region, if available"""
        if text_result := self.get("text_result"):
            return text_result.top_candidate()
        return None

    def create_segments(self, segments: Sequence[Segment]) -> None:
        """Segment this node"""
        self.children = [SegmentNode(segment, self) for segment in segments]

    def contains_text(self) -> bool:
        """Return True if this"""
        if self.text is not None:
            return True
        return any(child.contains_text() for child in self.children)

    def has_regions(self) -> bool:
        return all(not child.is_leaf() for child in self.children)

    def segments(self) -> "ImageGenerator":
        return ImageGenerator(self.leaves())

    def is_region(self) -> bool:
        return bool(self.children) and not self.text


class SegmentNode(ImageNode):
    """A node representing a segment of a page"""

    segment: Segment
    parent: ImageNode

    def __init__(self, segment: Segment, parent: ImageNode):
        bbox = segment.bbox.move(parent.coord)
        super().__init__(bbox.height, bbox.width, bbox.p1, segment.polygon, segment.mask, parent)
        self.add_data(segment=segment)
        self.segment = segment

    @property
    def image(self) -> "NamedImage":
        """The image this node represents"""
        bbox = self.segment.bbox
        mask = self.segment.mask
        img = imgproc.crop(self.parent.image, bbox)
        if mask is not None:
            img = imgproc.mask(img, mask)
        return NamedImage(img, self.label)


class PageNode(ImageNode):
    """A node representing a page / input image"""

    def __init__(self, image_path: str):
        self.path = image_path
        label = os.path.basename(image_path).split(".")[0]
        height, width = self.image.shape[:2]
        super().__init__(height, width, label=label)
        page_id = label.split("_")[-1]
        self.add_data(
            page_id=page_id,
            file_name=os.path.basename(image_path),
            image_path=image_path,
            image_name=label,
        )

    @property
    @lru_cache(maxsize=1)
    def image(self):
        return NamedImage(imgproc.read(self.path), self.label)


class Volume:

    """Class representing a collection of input images

    Examples:

    ```python
    from htrflow_core.volume import Volume

    images = ['../assets/demo_image.jpg'] * 5

    volume = Volume(images)
    ```

    """

    pages: list[PageNode]

    def __init__(self, paths: Iterable[str], label: str = "untitled_volume", label_format=None):
        """Initialize volume

        Arguments:
            paths: A list of paths to images
            label: A label describing the volume (optional)
        """
        pages = []
        for path in paths:
            try:
                page = PageNode(path)
            except imgproc.ImageImportError:
                logger.warning("Skipping %s (file format not supported)", path)
                continue
            pages.append(page)

        self.pages = pages
        self.label = label
        self._label_format = label_format or {}
        logger.info("Initialized volume '%s' with %d pages", label, len(pages))

    def __iter__(self) -> Iterator[PageNode]:
        return iter(self.pages)

    @classmethod
    def from_directory(cls, path: str) -> "Volume":
        """Initialize a volume from a directory

        Sets the volume label to the directory name.

        Arguments:
            path: A path to a directory of images.
        """
        files = (os.path.join(path, file) for file in sorted(os.listdir(path)))
        label = os.path.basename(path.strip("/"))
        return cls(files, label)

    @classmethod
    def from_pickle(cls, path: str) -> "Volume":
        """Initialize a volume from a pickle file

        Arguments:
            path: A path to a previously pickled volume instance
        """
        with open(path, "rb") as f:
            vol = pickle.load(f)

        if not isinstance(vol, Volume):
            raise pickle.UnpicklingError(f"Unpickling {path} did not return a Volume instance.")

        logger.info("Loaded volume '%s' from %s", vol.label, path)
        return vol

    def pickle(self, directory: str = ".cache", filename: Optional[str] = None):
        """Pickle volume

        Arguments:
            directory: Where to save the pickle file
            filename: Name of pickle file, optional. Defaults to
                <volume label>.pickle if left as None

        Returns:
            The path to the pickled file.
        """
        os.makedirs(directory, exist_ok=True)
        filename = f"{self.label}.pickle" if filename is None else filename
        path = os.path.join(directory, filename)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Wrote pickled volume '%s' to %s", self.label, path)
        return path

    def __str__(self):
        return f"Volume label: {self.label}\nVolume tree:\n" + "\n".join(child.tree2str() for child in self)

    def images(self) -> "ImageGenerator":
        """Yields the volume's original input images"""
        return ImageGenerator(page for page in self.pages)

    def segments(self) -> "ImageGenerator":
        """Yield the active segments' images"""
        return ImageGenerator(self.active_leaves())

    def leaves(self) -> Iterator[ImageNode]:
        yield from chain(page.leaves() for page in self)

    def active_leaves(self) -> Generator[ImageNode, None, None]:
        """Yield the volume's active leaves

        Here, an "active leaf" is a leaf node whose depth is equal to
        the maximum depth of the tree. In practice, this means that the
        node was segmented in the previous step (or is a fresh PageNode).
        Inactive leaves are leaves that weren't segmented in the
        previous step, and thus are higher up in the tree than the
        other leaves. These should typically not updated in the next
        steps.
        """
        max_depth = max(page.max_depth() for page in self)
        for leaf in self.leaves():
            if leaf.depth() == max_depth:
                yield leaf

    def update(self, results: list[Result]) -> None:
        """Update the volume with model results

        Arguments:
            results: A list of results where the i:th result
                corresponds to the volume's i:th active leaf node.
        """
        leaves = list(self.active_leaves())
        if len(leaves) != len(results):
            raise ValueError(f"Size of input ({len(results)}) does not match the size of the tree ({len(leaves)})")

        # Update the leaves of the tree
        for leaf, result in zip(leaves, results):
            # If the result has segments, segment the leaf
            if result.segments:
                leaf.create_segments(result.segments)

            # If the result has other data (e.g. texts), add it to the
            # new leaves (which may be other than `leaves` if the result
            # also had a segmentation)
            if result.data:
                for new_leaf, data in zip(leaf.leaves(), result.data):
                    new_leaf.add_data(**data)

        self.relabel()

    def save(self, directory: str = "outputs", serializer: str | serialization.Serializer = "alto") -> None:
        """Save volume

        Arguments:
            directory: Output directory
            serializer: What serializer to use, either a string name (e.g.,
                "alto") or a Serializer instance. See serialization.supported_formats()
                for available string options.
        """
        serialization.save_volume(self, serializer, directory)

    def set_label_format(self, **kwargs):
        self._label_format = kwargs

    def relabel(self):
        for page in self:
            page.relabel_levels(**self._label_format)


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


class NamedImage(np.ndarray):
    """An image (numpy array) with a `name` attribute

    This class is a thin wrapper around `np.ndarray` which adds a
    name attribute. It follows an example found in the numpy docs:
    https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """

    def __new__(cls, image: np.ndarray, name: str = "untitled_image"):
        obj = np.asarray(image).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)
