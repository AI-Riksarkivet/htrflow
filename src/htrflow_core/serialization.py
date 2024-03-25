from __future__ import annotations

import datetime
import json
import os
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

import xmlschema
from jinja2 import Environment, FileSystemLoader

import htrflow_core


if TYPE_CHECKING:
    from htrflow_core.volume import Node, PageNode, RegionNode, Volume


_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "templates/schema")


class Serializer:
    """Serializer base class.

    Each output format is implemented as a subclass to this class.

    Attributes:
        extension: The file extension assigned with this format, for
            example ".txt" or ".xml"
        format_name: The name of this format, for example "alto"
    """

    extension: str
    format_name: str

    def serialize(self, page: PageNode) -> str:
        """Serialize page

        Arguments:
            page: Input page

        Returns:
            A string"""

    def validate(self, doc: str):
        """Validate document"""


class AltoXML(Serializer):
    """Alto XML serializer"""

    extension = ".xml"
    format_name = "alto"

    def __init__(self):
        env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, "."]))
        self.template = env.get_template("alto")
        self.schema = os.path.join(_SCHEMA_DIR, "alto-4-4.xsd")

    def serialize(self, page: PageNode) -> str:
        # ALTO doesn't support nesting of regions ("TextBlock" elements)
        # This function is called from within the jinja template to tell
        # if a node corresponds to a TextBlock element, i.e. if its
        # children contains text and not other regions.
        def is_text_block(node):
            return bool(node.children) and all(child.is_line() for child in node.children)

        return self.template.render(
            page=page, metadata=metadata(page), labels=label_nodes(page), is_text_block=is_text_block
        )

    def validate(self, doc: str):
        xmlschema.validate(doc, self.schema)


class PageXML(Serializer):
    extension = ".xml"
    format_name = "page"

    def __init__(self):
        env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, "."]))
        self.template = env.get_template("page")
        self.schema = os.path.join(_SCHEMA_DIR, "pagecontent.xsd")

    def serialize(self, page: PageNode):
        return self.template.render(
            page=page,
            metadata=metadata(page),
            labels=label_nodes(page),
        )

    def validate(self, doc: str):
        xmlschema.validate(doc, self.schema)


class Json(Serializer):
    """Simple JSON serializer"""

    extension = ".json"
    format_name = "json"

    def __init__(self, include: Optional[Sequence[str]] = None):
        """Initialize JSON serializer

        Args:
            include: A list of attributes to include in the output JSON.
                If left as None, a default list of attributes is used,
                see `node2dict`.
        """
        self.include = include

    def serialize(self, page: PageNode):
        # Instead of passing `page` directly to json.dumps, it is first
        # converted to a dictionary with node2dict. This
        return json.dumps(node2dict(page, self.include), default=lambda o: getattr(o, "__dict__", None), indent=4)


class PlainText(Serializer):
    extension = ".txt"
    format_name = "txt"

    def serialize(self, page: PageNode) -> str:
        lines = page.traverse(lambda node: node.is_leaf())
        return "\n".join(line.text for line in lines)


def metadata(page: PageNode) -> dict[str, Union[str, list[dict[str, str]]]]:
    """Generate metadata for `page`

    Args:
        page: input page

    Returns:
        A dictionary with metadata
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    return {
        "creator": f"{htrflow_core.meta['Author']}",
        "software_name": f"{htrflow_core.meta['Name']}",
        "software_version": f"{htrflow_core.meta['Version']}",
        "application_description": f"{htrflow_core.meta['Summary']}",
        "created": timestamp,
        "last_change": timestamp,
        "processing_steps": [{"description": "", "settings": ""}],
    }


def supported_formats():
    """The supported formats"""
    return [cls.format_name for cls in Serializer.__subclasses__()]


def _get_serializer(format_name):
    for cls in Serializer.__subclasses__():
        if cls.format_name.lower() == format_name.lower():
            return cls()
    msg = f"Format '{format_name}' is not among the supported formats: {supported_formats()}"
    raise ValueError(msg)


def save_volume(volume: Volume, serializer: str | Serializer, dest: str) -> Iterable[tuple[str, str]]:
    """Serialize and save volume

    Arguments:
        volume: Input volume
        serializer: What serializer to use. Takes a Serializer instance
            or the name of the serializer as a string, see
            serialization.supported_formats() for supported formats.
        dest: Output directory
    """

    if isinstance(serializer, str):
        serializer = _get_serializer(serializer)

    dest = os.path.join(dest, volume.label)
    os.makedirs(dest, exist_ok=True)

    for page in volume:
        if not page.contains_text():
            raise ValueError(f"Cannot serialize page without text: {page.image_name}")

        doc = serializer.serialize(page)
        filename = os.path.join(dest, page.image_name + serializer.extension)

        with open(filename, "w") as f:
            f.write(doc)


def label_nodes(node: PageNode | RegionNode, template="%s") -> dict[PageNode | RegionNode, str]:
    """Assign labels to node and its descendants

    Arguments:
        node: Start node
        template: Label template
    """
    labels = {}
    labels[node] = template % node.label
    for i, child in enumerate(node.children):
        labels |= label_nodes(child, f"{labels[node]}_%s{i}")
    return labels


def node2dict(node: Node, include: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Convert node and its children to dictionary

    Recursively converts the tree starting at `node` to a dictionary.
    By default, this function includes the following attributes and
    properties: height, width, (global) bounding box, label, segment
    (i.e. the node's Segment instance, if available) and recognized text.

    Arguments:
        node: Starting node
        include: Optional list of attributes and properties to include.
            If None, the default attributes are included. See `Node` and
            `BaseDocumentNode` for available attributes and properties.

    Returns:
        A dictionary containing the node's data.
    """
    d = {}

    # Default attributes to include
    if include is None:
        include = ["height", "width", "bbox", "label", "recognized_text", "_segment"]

    for attrib in include:
        # Adds attribute to dictionary if 1) it exists and 2) it is "True",
        # i.e. it is not an empty list or string
        if val := getattr(node, attrib, False):
            d[attrib] = val

    if node.children:
        d["children"] = [node2dict(child, include) for child in node.children]

    return d
