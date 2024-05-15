"""
This module holds the base data structures
"""
import logging
import os
import pickle
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np

from htrflow_core import serialization
from htrflow_core.results import Result, Segment
from htrflow_core.utils import draw, imgproc
from htrflow_core.utils.geometry import Bbox, Point, Polygon
from htrflow_core.volume import node


logger = logging.getLogger(__name__)


class BaseDocumentNode(node.Node):
    """Extension of Node class with functionality related to documents"""

    _height: int
    _width: int
    _coord: Point

    def __str__(self) -> str:
        s = f"{self.height}x{self.width} node ({self.label}) at ({self.coord.x}, {self.coord.y})"
        if self.text:
            s += f": {self.text}"
        return s

    @property
    def image(self):
        """Image of the region this node represents"""
        return None

    @property
    def text(self) -> str | None:
        """Text of this region, if available"""
        if text_result := self.get("text_result"):
            return text_result.top_candidate()
        return None

    @property
    def height(self) -> int:
        """Height of the region this node represents"""
        return self._height

    @property
    def width(self) -> int:
        """Width of the region this node represents"""
        return self._width

    @property
    def coord(self) -> Point:
        """
        Position of the region this node represents relative to the
        original input image (root node of the tree).
        """
        if self.parent:
            return self._coord.move(self.parent.coord)
        return Point(0, 0)

    @property
    def polygon(self) -> Polygon:
        """
        Approximation of the mask of the region this node represents
        relative to the original input image (root node of the tree).
        If no mask is available, this attribute defaults to a polygon
        representation of the region's bounding box.
        """
        return self.bbox.polygon()

    @property
    def bbox(self) -> Bbox:
        """
        Bounding box of the region this node represents relative to
        the original input image (root node of the tree).
        """
        x, y = self.coord
        return Bbox(x, y, x + self.width, y + self.height)

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
        self._segment = segment
        self._height = segment.bbox.height
        self._width = segment.bbox.width
        self._coord = segment.bbox.p1
        self.add_data(
            segment=segment,
        )

    @property
    def polygon(self):
        return self._segment.polygon.move(self.parent.coord) or self.bbox.polygon()

    @property
    def image(self):
        """The image this node represents"""
        bbox = self._segment.bbox
        mask = self._segment.mask
        img = imgproc.crop(self.parent.image, bbox)
        if mask is not None:
            img = imgproc.mask(img, mask)
        return NamedImage(img, f"{self.get('long_label')}")


class PageNode(BaseDocumentNode):
    """A node representing a page / input image"""

    def __init__(self, image_path: str):
        super().__init__()
        self.path = image_path
        # Extract image name and remove file extension (`path/to/image.jpg` -> `image`)
        name = os.path.basename(image_path).split(".")[0]
        page_id = name.split("_")[-1]
        self.add_data(
            page_id=page_id,
            file_name=os.path.basename(image_path),
            image_path=image_path,
            image_name=name,
            label=name,
        )
        height, width = self.image.shape[:2]
        self._height = height
        self._width = width

    @property
    @lru_cache(maxsize=1)
    def image(self):
        return NamedImage(imgproc.read(self.path), self.get("long_label"))

    @image.setter
    def image(self, image):
        self._image = image

    def visualize(self, dest: Optional[str] = None, labels: str = "label", max_levels: int = 2):
        """Visualize the page

        Draws the page's regions on its image.

        Arguments:
            dest: An optional path where to save the resulting image.
            labels: Labels to annotate the drawn regions. Defaults to
                "label", i.e. the node's label.
            max_levels: How many levels of segmentation to draw.
                Defaults to 2.

        Returns:
            An annotated image.
        """
        regions = self.traverse(lambda node: node != self)
        depths = {region.depth() for region in regions}
        colors = [draw.Colors.GREEN, draw.Colors.RED, draw.Colors.BLUE]

        image = self.image
        for depth in sorted(depths)[:max_levels]:
            group = [region for region in regions if region.depth() == depth]
            labels_ = [region.get(labels, "") for region in group]
            polygons = [region.polygon for region in group]
            image = draw.draw_polygons(image, polygons, labels=labels_, color=colors[depth % 3])

        if dest is not None:
            imgproc.write(dest, image)
        return image


class Volume(BaseDocumentNode):

    """Class representing a collection of input images

    Examples:

    ```python
    from htrflow_core.volume import Volume

    images = ['../assets/demo_image.jpg'] * 5

    volume = Volume(images)
    ```

    """

    def __init__(self, paths: Iterable[str], label: str = "untitled_volume", label_format={}):
        """Initialize volume

        Arguments:
            paths: A list of paths to images
            label: A label describing the volume (optional)
        """
        super().__init__()
        for path in paths:
            try:
                page = PageNode(path)
            except imgproc.ImageImportError:
                logger.warn("Skipping %s (file format not supported)", path)
                continue
            self.children.append(page)

        self._label_format = label_format
        self.add_data(label=label)
        logger.info("Initialized volume '%s' with %d pages", label, len(self.children))

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
        return f"Volume label: {self.get('label')}\nVolume tree:\n" + "\n".join(child.tree2str() for child in self)

    def images(self) -> "ImageGenerator":
        """Yields the volume's original input images"""
        return ImageGenerator(self.children)

    def segments(self) -> "ImageGenerator":
        """Yield the active segments' images"""
        return ImageGenerator(self.active_leaves())

    def active_leaves(self):
        """Yield the volume's active leaves

        Here, an "active leaf" is a leaf node whose depth is equal to
        the maximum depth of the tree. In practice, this means that the
        node was segmented in the previous step (or is a fresh PageNode).
        Inactive leaves are leaves that weren't segmented in the
        previous step, and thus are higher up in the tree than the
        other leaves. These should typically not updated in the next
        steps.
        """
        max_depth = self.max_depth()
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
            raise ValueError(f"Size of input ({len(results)}) does not match " f"the size of the tree ({len(leaves)})")

        # Update the leaves of the tree
        for leaf, result in zip(leaves, results):
            # If the result has segments, segment the leaf
            if result.segments:
                leaf.segment(result.segments)

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

    def __init__(self, nodes: Sequence[node.Node]):
        self._nodes = list(nodes)

    def __iter__(self):
        for _node in self._nodes:
            yield _node.image

    def __len__(self):
        return len(self._nodes)


class NamedImage(np.ndarray):
    """An image (numpy array) with a `name` attribute

    This class is a thin wrapper around `np.ndarray` which adds a
    name attribute. It follows an example found in the numpy docs:
    https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """

    def __new__(cls, image, name="untitled_image"):
        obj = np.asarray(image).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)
