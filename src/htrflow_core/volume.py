"""
This module holds the base data structures
"""

import os
import pickle
from abc import ABC, abstractproperty
from copy import deepcopy
from typing import Any, Callable, Iterable, Optional, Sequence

from htrflow_core import serialization
from htrflow_core.results import RecognizedText, Result, Segment
from htrflow_core.serialization import Serializer
from htrflow_core.utils import imgproc
from htrflow_core.utils.geometry import Bbox, Point, Polygon


class Node:
    """Node class"""

    parent: Optional["Node"]
    children: Sequence["Node"]
    data: dict[str, Any]

    def __init__(self, parent: Optional["Node"] = None):
        self.parent = parent
        self.children = []
        self.data = {}

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.children[i]
        i, *rest = i
        return self.children[i][rest] if rest else self.children[i]

    def __iter__(self):
        return iter(self.children)

    def depth(self):
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

    def add_data(self, **data):
        self.data |= data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def leaves(self):  # -> Sequence[Self]:
        """Return the leaf nodes attached to this node"""
        return self.traverse(filter=lambda node: node.is_leaf())

    def traverse(self, filter: Optional[Callable[["Node"], bool]] = None) -> Sequence["Node"]:
        """Return all nodes attached to this node"""
        nodes = [self] if (filter is None or filter(self)) else []
        for child in self.children:
            nodes.extend(child.traverse(filter=filter))
        return nodes

    def tree2str(self, sep: str = "", is_last: bool = True) -> str:
        """Return a string representation of this node and its decendants"""
        lines = [sep + ("└──" if is_last else "├──") + str(self)]
        sep += "    " if is_last else "│   "
        for child in self.children:
            lines.append(child.tree2str(sep, child == self.children[-1]))
        return "\n".join(lines).strip("└──")

    def is_leaf(self) -> bool:
        """True if this node does not have any children"""
        return not self.children

    def asdict(self):
        """This node's and its decendents' data as a dictionary"""
        if self.is_leaf():
            return self.data
        return self.data | {"contains": [child.asdict() for child in self.children]}

    def detach(self):
        """Detach node from tree

        Removes the node from its parent's children and sets its parent
        to None, effectively removing it from the tree.
        """
        if self.parent:
            siblings = self.parent.children
            self.parent.children = [child for child in siblings if child != self]
        self.parent = None

    def prune(self, condition: Callable[["Node"], bool], include_starting_node=True):
        """Prune the tree

        Removes (detaches) all nodes starting from this node that
        fulfil the given condition. Any decendents of a node that
        fulfils the condition are also removed.

        Arguments:
            condition: A function `f` where `f(node) == True` if `node`
                should be removed from the tree.
            include_starting_node: Whether to include the starting node
                or not. If False, the starting node will not be
                detached from its parent even though it fulfils the
                given condition. Defaults to True.

        Example: To remove all nodes at depth 2, use
            node.prune(lambda node: node.depth() == 2)
        """
        nodes = self.traverse(filter=condition)
        for node in nodes:
            if not include_starting_node and node == self:
                continue
            node.detach()


class BaseDocumentNode(Node, ABC):
    """Extension of Node class with functionality related to documents"""

    def __str__(self) -> str:
        s = f"{self.height}x{self.width} node ({self.label}) at ({self.coord.x}, {self.coord.y})"
        if self.text:
            s += f": {self.text}"
        return s

    @abstractproperty
    def image(self):
        """Image of the region this node represents"""

    @property
    def text(self) -> str | None:
        """Text of this region, if available"""
        if text_result := self.get("text_result"):
            return text_result.top_candidate()
        return None

    @property
    def height(self) -> int:
        """Height of the region this node represents"""
        return self.get("height")

    @property
    def width(self) -> int:
        """Width of the region this node represents"""
        return self.get("width")

    @property
    def coord(self) -> Point:
        """
        Position of the region this node represents relative to the
        original input image (root node of the tree).
        """
        return self.get("coord", Point(0, 0))

    @property
    def polygon(self) -> Polygon:
        """
        Approximation of the mask of the region this node represents
        relative to the original input image (root node of the tree).
        If no mask is available, this attribute defaults to a polygon
        representation of the region's bounding box.
        """
        return self.get("polygon")

    @property
    def bbox(self) -> Bbox:
        """
        Bounding box of the region this node represents relative to
        the original input image (root node of the tree).
        """
        x, y = self.coord
        return Bbox(x, y, x + self.width, y + self.height)

    def add_text(self, text: RecognizedText):
        self.add_data(text_result = text)

    @property
    def label(self):
        return self.get("label", "node")

    def segment(self, segments: Sequence[Segment]):
        """Segment this node"""
        children = []
        for segment in segments:
            children.append(RegionNode(segment, self))
        self.children = children

    def contains_text(self) -> bool:
        if self.text is not None:
            return True
        return any(child.contains_text() for child in self.children)

    def has_regions(self) -> bool:
        return all(not child.is_leaf() for child in self.children)

    def segments(self):
        return ImageGenerator(self.leaves())

    def is_region(self) -> bool:
        return bool(self.children) and not self.text


class RegionNode(BaseDocumentNode):
    """A node representing a segment of a page"""

    def __init__(self, segment: Segment, parent: BaseDocumentNode):
        super().__init__(parent=parent)
        self.add_data(
            height=segment.bbox.height,
            width=segment.bbox.width,
            polygon=segment.polygon.move(parent.coord),
            coord=segment.bbox.p1.move(parent.coord),
            segment=segment,
        )

    @property
    def image(self):
        """The image this node represents"""
        segment = self.get("segment")
        bbox = segment.bbox
        mask = segment.mask
        img = imgproc.crop(self.parent.image, bbox)
        if mask is not None:
            img = imgproc.mask(img, mask)
        return img


class PageNode(BaseDocumentNode):
    """A node representing a page / input image"""

    def __init__(self, image_path: str):
        super().__init__()
        self._image = imgproc.read(image_path)
        # Extract image name and remove file extension (`path/to/image.jpg` -> `image`)
        name = os.path.basename(image_path).split(".")[0]
        height, width = self.image.shape[:2]
        self.add_data(
            image_path = image_path,
            image_name = name,
            height = height,
            width = width,
            polygon = Bbox(0, 0, width, height).polygon(),
            label = name
        )

    @property
    def image(self):
        return self._image


class Volume(Node):

    """Class representing a collection of input images

    Examples:

    ```python
    from htrflow_core.volume import Volume

    images = ['../assets/demo_image.jpg'] * 5

    volume = Volume(images)
    ```

    """

    def __init__(self, paths: Iterable[str], label: str = "untitled_volume"):
        """Initialize volume

        Arguments:
            paths: A list of paths to images
            label: A label describing the volume (optional)
        """
        super().__init__()
        self.children = [PageNode(path) for path in paths]
        self.add_data(label=label)
        self.label = label

    @classmethod
    def from_directory(cls, path: str) -> "Volume":
        """Initialize a volume from a directory

        Sets the volume label to the directory name.

        Arguments:
            path: A path to a directory of images.
        """
        files = (os.path.join(path, file) for file in sorted(os.listdir(path)))
        label = os.path.basename(path)
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

        return vol

    def pickle(self, directory: str = ".cache", filename: Optional[str] = None):
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
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    def __str__(self):
        return f"Volume label: {self.get('label')}\nVolume tree:\n" + "\n".join(child.tree2str() for child in self)

    def images(self):  # -> Generator[np.ndarray]:
        """Yields the volume's original input images"""
        return ImageGenerator(self.children)

    def segments(self, depth: Optional[int] = None):  # -> Iterable[np.ndarray]:
        """Yields the volume's segments at `depth`

        Args:
            depth (int | None): Which depth segments to yield. Defaults to None, which
                returns the leaf nodes (maximum depth).
        """
        if depth is None:
            return ImageGenerator(self.leaves())

        else:
            filtered_nodes = (node for node in self.traverse() if node.depth == depth)
            return ImageGenerator(filtered_nodes)

    def update(self, results: list[Result]) -> None:
        """
        Update the volume with model results

        Arguments:
            results: A list of results where the i:th result corresponds
                to the volume's i:th leaf node.
        """
        leaves = list(self.leaves())
        if len(leaves) != len(results):
            raise ValueError(f"Size of input ({len(results)}) does not match " f"the size of the tree ({len(leaves)})")

        # Update the leaves of the tree
        for leaf, result in zip(leaves, results):
            # If the result has segments, segment the leaf
            if result.segments:
                leaf.segment(result.segments)

            # If the result has texts, add them to the new leaves (which
            # may be other than `leaves` if the result also had a segmentation)
            if result.texts:
                for new_leaf, text in zip(leaf.leaves(), result.texts):
                    new_leaf.add_text(text)

    def save(self, directory: str = "outputs", serializer: str | Serializer = "alto") -> None:
        """Save volume

        Arguments:
            directory: Output directory
            serializer: What serializer to use, either a string name (e.g.,
                "alto") or a Serializer instance. See serialization.supported_formats()
                for available string options.
        """
        serialization.save_volume(self, serializer, directory)


class ImageGenerator:
    """A generator with __len__

    Wrapper around `nodes` that provides a generator over the nodes'
    images and implements len(). This way, there is no need to load
    all images into memory at once, but the length of the generator
    is known beforehand (which is typically not the case), which is
    handy in some cases, e.g., when using tqdm progress bars.
    """
    def __init__(self, nodes: Sequence[Node]):
        self._nodes = list(nodes)

    def __iter__(self):
        for node in self._nodes:
            yield node.image

    def __len__(self):
        return len(self._nodes)


def remove_noise_regions(volume: Volume, threshold: float = 0.8):
    """Remove noise regions from volume

    Makes a copy of the given volume where noisy regions are removed.
    Uses the heuristic defined in `is_noise`.

    Arguments:
        volume: Input volume with text and regions
        threshold: The confidence score threshold, default 0.8.

    Returns:
        A copy of `volume` where all regions have an average text
        recognition confidence score above the given threshold.
    """
    volume = deepcopy(volume)
    volume.prune(lambda node: is_noise(node, threshold), include_starting_node=False)
    return volume


def is_noise(node: BaseDocumentNode, threshold: float = 0.8):
    """Heuristically determine if region is noise

    Assumes that a region is noise if the average text recognition
    confidence score is lower than the given threshold.

    Arguments:
        node: Which node to check
        threshold: Threshold for the average text recognition
            confidence score. Defaults to 0.8, i.e., any region with
            avg. confidence lower than 0.8 is regarded as noise.

    Returns:
        True if `node` is a region (i.e. parent to nodes with text lines)
        and the average text recognition confidence score of its
        children is below `threshold`.
    """
    if node.children and all(child.is_line() for child in node):
        conf = sum(child.get("text_result").top_score() for child in node) / len(node.children)
        return conf < threshold
    return False
