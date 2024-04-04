"""
This module holds the base data structures
"""

import logging
import os
import pickle
from typing import Iterable, Optional, Sequence

from htrflow_core import serialization
from htrflow_core.results import Result, Segment
from htrflow_core.utils import draw, imgproc
from htrflow_core.utils.geometry import Bbox, Point, Polygon
from htrflow_core.volume import node


logger = logging.getLogger(__name__)


class BaseDocumentNode(node.Node):
    """Extension of Node class with functionality related to documents"""

    def __str__(self) -> str:
        s = f"{self.height}x{self.width} node ({self.label}) at ({self.coord.x}, {self.coord.y})"
        if self.text:
            s += f": {self.text}"
        return s

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

    @property
    def label(self):
        return self.get("label", "node")

    def segment(self, segments: Sequence[Segment]):
        """Segment this node"""
        children = []
        for segment in segments:
            children.append(RegionNode(segment, self))
        self.children = children
        logger.info("Created %d new nodes", len(children))

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

    def visualize(self, dest: Optional[str]=None, labels: str="label", max_levels: int=2):
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

    def __init__(self, paths: Iterable[str], label: str = "untitled_volume"):
        """Initialize volume

        Arguments:
            paths: A list of paths to images
            label: A label describing the volume (optional)
        """
        super().__init__()
        self.children = [PageNode(path) for path in paths]
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

        logger.info("Loaded volume '%s' from %s", vol.label, path)
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
        logger.info("Wrote pickled volume '%s' to %s", self.label, path)
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

            # If the result has other data (e.g. texts), add it to the
            # new leaves (which may be other than `leaves` if the result
            # also had a segmentation)
            if result.data:
                for new_leaf, data in zip(leaf.leaves(), result.data):
                    new_leaf.add_data(**data)

    def save(self, directory: str = "outputs", serializer: str | serialization.Serializer = "alto") -> None:
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
    def __init__(self, nodes: Sequence[node.Node]):
        self._nodes = list(nodes)

    def __iter__(self):
        for node in self._nodes:
            yield node.image

    def __len__(self):
        return len(self._nodes)
