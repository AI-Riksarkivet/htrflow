"""
This module holds the base data structures
"""

import os
from collections import defaultdict
from functools import singledispatchmethod
from itertools import count
from typing import Literal, Optional

import cv2
import numpy as np

from htrflow_core import image, serialization
from htrflow_core.results import RecognitionResult, Segment, SegmentationResult


class Node:

    """Node class"""

    # Page/image-related stuff
    image: np.ndarray
    height: int
    width: int
    x: int
    y: int

    # Node/tree-related stuff
    parent: Optional["Node"]
    children: list["Node"]
    depth: int
    _id_generator = defaultdict(lambda: count(0))

    def __init__(self, parent: "Node" = None, x: int = 0, y: int = 0):
        self.parent = parent
        self.children = []
        self.depth = parent.depth + 1 if parent else 0
        self.x = parent.x + x if parent else x
        self.y = parent.y + y if parent else y
        self.id_ = next(Node._id_generator[parent])
        self.text = None

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.children[i]
        i, *rest = i
        return self.children[i][rest] if rest else self.children[i]

    def leaves(self):
        """Return the leaf nodes attached to this node"""
        nodes = [] if self.children else [self]
        for child in self.children:
            nodes.extend(child.leaves())
        return nodes

    def traverse(self):
        """Return all nodes attached to this node"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.traverse())
        return nodes

    def lines(self):
        """Return all nodes that contains text attached to this node"""
        return [node for node in self.traverse() if node.text]

    def tree2str(self, sep="", is_last=True):
        """Return a string representation of this node and its decendants"""
        lines = [sep + ("└──" if is_last else "├──") + str(self)]
        sep += "    " if is_last else "│   "
        for child in self.children:
            lines.append(child.tree2str(sep, child == self.children[-1]))
        return "\n".join(lines)

    @singledispatchmethod
    def update(self, result: SegmentationResult | RecognitionResult):
        """Update this node with `result`"""

    @update.register
    def _(self, result: SegmentationResult):
        """Segment this node

        Creates children according to the segmentation and attaches them to this node.
        """
        children = []
        for segment in result.segments:
            children.append(RegionNode(segment, self))
        self.children = children

    @update.register
    def _(self, result: RecognitionResult):
        """Update the text of this node"""
        self.text = result


class RegionNode(Node):
    """A node representing a segment of a page"""

    def __init__(self, segment: Segment, parent: Node):
        self.segment = segment
        x, _, y, _ = segment.bbox
        super().__init__(parent, x, y)

    def __str__(self):
        s = f'{self.height}x{self.width} region at ({self.x}, {self.y})'
        if self.text:
            s += f' "{self.text.top_candidate()}"'
        return s

    @property
    def image(self):
        """The image this segment represents"""
        bbox = self.segment.bbox
        mask = self.segment.mask
        return image.mask(image.crop(self.parent.image, bbox), mask)

    @property
    def height(self) -> int:
        """Height of region"""
        *_, y1, y2 = self.segment.bbox
        return y2-y1

    @property
    def width(self) -> int:
        """Width of region"""
        x1, x2, *_ = self.segment.bbox
        return x2-x1

    @property
    def polygon(self):
        """Region polygon, relative to original image size"""
        return [(self.parent.x + x, self.parent.y + y) for x,y in self.segment.polygon]

    @property
    def bbox(self):
        """Region bounding box, relative to original image size"""
        x, y = self.parent.x, self.parent.y
        x1, x2, y1, y2 = self.segment.bbox
        return [x1+x, x2+x, y1+y, y2+y]


class PageNode(Node):

    """A node representing a page / input image"""

    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.image_path = image_path
        # Extract image name and remove file extension (`path/to/image.jpg` -> `image`)
        self.image_name = os.path.basename(image_path).split(".")[0]
        super().__init__()

    def __str__(self):
        return f'{self.height}x{self.width} image {self.image_name}'

    @property
    def height(self):
        """Image height"""
        return self.image.shape[0]

    @property
    def width(self):
        """Image width"""
        return self.image.shape[1]

    @property
    def polygon(self):
        return self.bbox

    @property
    def bbox(self):
        return [(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)]

    def show(self):
        polygons = [np.array(node.polygon) for node in self.traverse() if node != self]
        im = image.draw_polygons(self.image, polygons)
        cv2.imshow('image', im)
        cv2.waitKey(0)


class Volume:

    """Class representing a collection of input images"""

    def __init__(self, paths: list[str]):
        self._root = Node()
        self._root.children = [PageNode(path) for path in paths]

    def __getitem__(self, i):
        return self._root[i]

    def __iter__(self):
        return self._root.children.__iter__()

    def __str__(self):
        return self._root.tree2str()

    def images(self):  # -> Generator[np.ndarray]:
        """Yields the volume's original input images"""
        for image_node in self._root.children:
            yield image_node.image

    def segments(self, depth: int = None):  # -> Generator[np.ndarray]:
        """Yields the volume's segments at `depth`

        Args:
            depth (int | None): Which depth segments to yield. Defaults to None, which
                returns the leaf nodes (maximum depth).
        """

        if depth is None:
            for segment in self._root.leaves():
                yield segment.image
        else:
            for segment in self._root.traverse():
                if segment.depth == depth:
                    yield segment.image

    def update(self, results: list[SegmentationResult | RecognitionResult]):
        """Update the volume with segmentation or text recognition results"""
        leaves = self._root.leaves()
        if len(leaves) != len(results):
            raise ValueError(f"Size of input ({len(results)}) does not match "
                             f"the size of the tree ({len(leaves)})")

        for node, result in zip(leaves, results):
            node.update(result)

        return self.segments()

    def save(self, directory='outputs', format_: Literal['alto', 'page', 'txt'] = 'alto'):
        """Save volume

        Arguments:
            directory: Output directory
            format_: Output format
        """
        serializer = serialization.get_serializer(format_)
        pages = serializer(self)
        if pages:
            os.makedirs(directory, exist_ok=True)
            for page, filename in pages:
                with open(os.path.join(directory, filename), 'w') as f:
                    f.write(page)
