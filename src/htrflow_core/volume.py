"""
This module holds the base data structures
"""

import os
import pickle
from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from itertools import chain
from typing import Callable, Iterable, Literal, Optional, Sequence

import cv2

from htrflow_core import image, serialization
from htrflow_core.results import RecognizedText, Result, Segment


Point = namedtuple("Point", ["x", "y"])


class Node:
    """Node class"""

    parent: Optional["Node"]
    children: Sequence["Node"]
    depth: int

    def __init__(self, parent: Optional["Node"] = None):
        self.parent = parent
        self.children = []
        self.depth = parent.depth + 1 if parent else 0

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.children[i]
        i, *rest = i
        return self.children[i][rest] if rest else self.children[i]

    def leaves(self): # -> Sequence[Self]:
        """Return the leaf nodes attached to this node"""
        nodes = [] if self.children else [self]
        for child in self.children:
            nodes.extend(child.leaves())
        return nodes

    def traverse(self, filter: Optional[Callable[["Node"], bool]] = None) -> Sequence["Node"]:
        """Return all nodes attached to this node"""
        nodes = [self] if (filter is None or filter(self)) else []
        for child in self.children:
            nodes.extend(child.traverse(filter=filter))
        return nodes

    def tree2str(self, sep: str="", is_last: bool=True) -> str:
        """Return a string representation of this node and its decendants"""
        lines = [sep + ("└──" if is_last else "├──") + str(self)]
        sep += "    " if is_last else "│   "
        for child in self.children:
            lines.append(child.tree2str(sep, child == self.children[-1]))
        return "\n".join(lines).strip("└──")

    def is_leaf(self) -> bool:
        return not self.children


class BaseDocumentNode(Node, ABC):
    """Extension of Node class with functionality related to documents"""

    height: int
    width: int
    coord: Point
    label: str
    polygon: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    children: Sequence["RegionNode"]
    _text: Optional[RecognizedText]
    text: Optional[str]

    def __str__(self) -> str:
        return f'{self.height}x{self.width} region ({self.label}) at ({self.coord.x}, {self.coord.y})'

    @abstractproperty
    def image(self):
        pass

    def add_text(self, text: RecognizedText):
        """Add text to this node"""
        self._text = text
        self.text = text.top_candidate()

    def segment(self, segments: Sequence[Segment]):
        """Segment this node"""
        children = []
        for segment in segments:
            children.append(RegionNode(segment, self))
        self.children = children

    def contains_text(self) -> bool:
        return any(child.contains_text() for child in self.children)

    def has_regions(self) -> bool:
        return all(not child.is_leaf() for child in self.children)

    def segments(self):
        for leaf in self.leaves():
            yield leaf.image


class RegionNode(BaseDocumentNode):
    """A node representing a segment of a page"""

    _segment: Segment
    parent: BaseDocumentNode

    DEFAULT_LABEL = "region"

    def __init__(self, segment: Segment, parent: BaseDocumentNode):
        super().__init__(parent)
        self._segment = segment
        self.text = None
        self.label = segment.class_label if segment.class_label else RegionNode.DEFAULT_LABEL
        x1, x2, y1, y2 = segment.bbox
        self.height = y2 - y1
        self.width = x2 - x1
        self.polygon = [(x + parent.coord.x, y + parent.coord.y) for x, y in segment.polygon]
        self.bbox = (x1 + parent.coord.x, x2 + parent.coord.x, y1 + parent.coord.y, y2 + parent.coord.y)
        self.coord = Point(parent.coord.x + x1, parent.coord.y + y1)

    def __str__(self) -> str:
        if self.text:
            return f'{super().__str__()}: "{self.text}"'
        return super().__str__()

    @property
    def image(self):
        """The image this segment represents"""
        img = image.crop(self.parent.image, self._segment.bbox)
        if self._segment.mask is not None:
            img = image.mask(img, self._segment.mask)
        return img

    def contains_text(self) -> bool:
        if not self.children:
            return self.text is not None
        return super().contains_text()

    def is_region(self) -> bool:
        return bool(self.children) and not self.text

    def is_word(self) -> bool:
        return self.text is not None and len(self.text.split()) == 1

    def is_line(self):
        return self.text is not None and len(self.text.split()) > 1


class PageNode(BaseDocumentNode):
    """A node representing a page / input image"""

    text = None

    def __init__(self, image_path: str):
        self._image = cv2.imread(image_path)
        self.image_path = image_path
        self.height, self.width = self.image.shape[:2]
        self.coord = Point(0, 0)
        self.polygon = [(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)]
        self.bbox = (0, self.width, 0, self.height)

        # Extract image name and remove file extension (`path/to/image.jpg` -> `image`)
        self.image_name = os.path.basename(image_path).split(".")[0]
        self.label = self.image_name
        super().__init__()

    @property
    def image(self):
        return self._image

class Volume:

    """Class representing a collection of input images"""

    def __init__(self, paths: Iterable[str], label: str="untitled_volume"):
        """Initialize volume

        Arguments:
            paths: A list of paths to images
            label: A label describing the volume (optional)
        """
        self.pages = [PageNode(path) for path in paths]
        self.label = label

    @classmethod
    def from_directory(cls, path: str) -> "Volume":
        """Initialize a volume from a directory

        Sets the volume label to the directory name.

        Arguments:
            path: A path to a directory of images.
        """
        files = (os.path.join(path, file) for file in os.listdir(path))
        label = os.path.basename(path)
        return cls(files, label)

    @classmethod
    def from_pickle(cls, path: str) -> "Volume":
        """Initialize a volume from a pickle file

        Arguments:
            path: A path to a previously pickled volume instance
        """
        with open(path, 'rb') as f:
            vol = pickle.load(f)
        return vol

    def pickle(self, directory: str=".cache", filename: Optional[str] = None):
        """Pickle volume

        Arguments:
            directory: Where to save the pickle file
            filename: Name of pickle file, optional. Defaults to
                "volume_{volume label}.pickle" if left as None

        Returns:
            The path to the pickled file.
        """
        os.makedirs(directory, exist_ok=True)
        filename = f"volume_{self.label}.pickle" if filename is None else filename
        path = os.path.join(directory, filename)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return path

    def __getitem__(self, i):
        if isinstance(i, Iterable):
            i, *rest = i
            return self.pages[i].__getitem__(rest)
        return self.pages[i]

    def __iter__(self):
        return self.pages.__iter__()

    def __str__(self):
        return f"Volume label: {self.label}\nVolume tree:\n" + "\n".join(page.tree2str() for page in self.pages)

    def images(self):  # -> Generator[np.ndarray]:
        """Yields the volume's original input images"""
        for page in self.pages:
            yield page.image

    def leaves(self): # -> Iterable[BaseDocumentNode]:
        return chain.from_iterable(page.leaves() for page in self.pages)

    def traverse(self):
        return chain.from_iterable(page.traverse() for page in self.pages)

    def segments(self, depth: Optional[int] = None): # -> Iterable[np.ndarray]:
        """Yields the volume's segments at `depth`

        Args:
            depth (int | None): Which depth segments to yield. Defaults to None, which
                returns the leaf nodes (maximum depth).
        """

        if depth is None:
            for node in self.leaves():
                yield node.image
        else:
            for page in self.pages:
                for node in page.traverse(lambda node: node.depth == depth):
                    yield node.image

    def update(self, results: list[Result]) -> None:
        """
        Update the volume with model results

        Arguments:
            results: A list of results where the i:th result corresponds
                to the volume's i:th leaf node.
        """
        leaves = list(self.leaves())
        if len(leaves) != len(results):
            raise ValueError(f"Size of input ({len(results)}) does not match "
                             f"the size of the tree ({len(leaves)})")

        # Update the leaves of the tree
        for leaf, result in zip(leaves, results):

            # If the result has segments, segment the leaf
            if result.segments:
                leaf.segment(result.segments)

            # If the result has texts, add them to the new leaves (which
            # may be other than `leaves` if the result also had a segmentation)
            if result.texts:
                print(result.texts, type(result.texts))
                for new_leaf, text in zip(leaf.leaves(), result.texts):
                    new_leaf.add_text(text)

    def save(self, directory: str="outputs", format_: Literal["alto", "page", "txt"] = "alto") -> None:
        """Save volume

        Arguments:
            directory: Output directory
            format_: Output format
        """
        serialization.save_volume(self, format_, directory)
