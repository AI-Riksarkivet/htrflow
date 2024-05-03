from __future__ import annotations

import datetime
import json
import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Sequence, Union

import xmlschema
from jinja2 import Environment, FileSystemLoader

import htrflow_core
from htrflow_core.utils.layout import RegionLocation


if TYPE_CHECKING:
    from htrflow_core.volume import PageNode, Volume


logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "templates/schema")


class Serializer:
    """Serializer base class.

    Each output format is implemented as a subclass to this class.

    Attributes:
        extension: The file extension associated with this format, for
            example ".txt" or ".xml"
        format_name: The name of this format, for example "alto".
    """

    extension: str
    format_name: str

    def serialize(self, page: PageNode, validate=False) -> str:
        """Serialize page

        Arguments:
            page: Input page
            validate: If True, the generated document is passed through
                valiadation before return.

        Returns:
            A string"""
        doc = self._serialize(page)
        if validate:
            self.validate(doc)
        return doc

    def serialize_volume(self, volume: Volume) -> Sequence[tuple[str, str]]:
        """Serialize volume

        Arguments:
            volume: Input volume

        Returns:
            A sequence of (document, filename) tuples where `document`
            is the serialized version of volume and `filename` is a
            suggested filename to save `document` to. Note that this
            method may produce one file (which covers the entire
            volume) or several files (typically one file per page),
            depending on the serialization method.
        """
        outputs = []
        for page in volume:
            doc = self.serialize(page)
            if doc is None:
                continue
            filename = os.path.join(volume.label, page.label + self.extension)
            outputs.append((doc, filename))
        return outputs

    def validate(self, doc: str):
        """Validate document"""

    def _serialize(self, page: PageNode) -> str:
        """Format-specific seralization method"""


class AltoXML(Serializer):
    """Alto XML serializer"""

    extension = ".xml"
    format_name = "alto"

    def __init__(self):
        env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, "."]))
        self.template = env.get_template("alto")
        self.schema = os.path.join(_SCHEMA_DIR, "alto-4-4.xsd")

    def _serialize(self, page: PageNode) -> str:
        # Find all nodes that correspond to Alto TextBlock elements and
        # their location (if available). A TextBlock is a region whose
        # children are text lines (and not other regions). If the node's
        # `region_location` attribute is set, it will be rendered in the
        # corresponding Alto group, otherwise it will be rendered in the
        # printspace group.
        text_blocks = defaultdict(list)
        for node in page.traverse():
            if node.is_region() and all(child.text for child in node):
                text_blocks[node.get("region_location", RegionLocation.PRINTSPACE)].append(node)

        return self.template.render(
            page=page,
            printspace=text_blocks[RegionLocation.PRINTSPACE],
            top_margin=text_blocks[RegionLocation.MARGIN_TOP],
            bottom_margin=text_blocks[RegionLocation.MARGIN_BOTTOM],
            left_margin=text_blocks[RegionLocation.MARGIN_LEFT],
            right_margin=text_blocks[RegionLocation.MARGIN_RIGHT],
            metadata=metadata(page),
            xmlescape=xmlescape,
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

    def _serialize(self, page: PageNode):
        if page.is_leaf():
            return None

        def is_text_line(node):
            return node.text and (node.parent is None or node.parent.is_region())

        return self.template.render(
            page=page,
            metadata=metadata(page),
            is_text_line=is_text_line,
        )

    def validate(self, doc: str):
        xmlschema.validate(doc, self.schema)


class Json(Serializer):
    """Simple JSON serializer"""

    extension = ".json"
    format_name = "json"

    def __init__(self, one_file=False):
        """Initialize JSON serializer

        Args:
            one_file: Export all pages of the volume to the same file.
                Defaults to False.
        """
        self.one_file = one_file

    def _serialize(self, page: PageNode):
        def default(obj):
            return {k: v for k, v in obj.__dict__.items() if k not in ["mask", "_image", "parent"]}

        return json.dumps(page.asdict(), default=default, indent=4)

    def serialize_volume(self, volume: Volume):
        if self.one_file:
            filename = volume.label + self.extension
            doc = self.serialize(volume)
            return [(doc, filename)]
        return super().serialize_volume(volume)


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


def get_serializer(serializer_name, **serializer_args):
    for cls in Serializer.__subclasses__():
        if cls.format_name.lower() == serializer_name.lower():
            return cls(**serializer_args)
    msg = f"Format '{serializer_name}' is not among the supported formats: {supported_formats()}"
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
        serializer = get_serializer(serializer)
        logger.info("Using %s serializer with default settings", serializer.__class__.__name__)

    for doc, filename in serializer.serialize_volume(volume):
        filename = os.path.join(dest, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(doc)
        logger.info("Wrote document to %s", filename)


def xmlescape(s: str) -> str:
    """Escape special characters in XML strings

    Replaces the characters &, ", ', < and > with their corresponding
    character entity references.
    """
    s = s.replace("&", "&amp;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s
