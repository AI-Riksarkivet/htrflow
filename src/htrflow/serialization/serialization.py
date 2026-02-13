"""
Serialization module

This module contains functions for exporting a `Collection` or `PageNode`
instance to different formats.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import xmlschema
from jinja2 import Environment, FileSystemLoader

import htrflow
from htrflow.postprocess.metrics import average_text_confidence
from htrflow.results import TEXT_RESULT_KEY
from htrflow.utils.layout import REGION_KEY, RegionLocation


if TYPE_CHECKING:
    from htrflow.volume.volume import Collection, PageNode


logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")


class Serializer:
    """Serializer base class.

    Each output format is implemented as a subclass to this class.

    Attributes:
        extension: The file extension associated with this format, for
            example ".txt" or ".xml"
    """

    extension: str

    def serialize(self, page: PageNode, validate: bool = False, **metadata) -> str | None:
        """Serialize page

        Arguments:
            page: Input page
            validate: If True, the generated document is passed through
                valiadation before return.

        Returns:
            A string if the serialization was succesful, else None
        """
        doc = self._serialize(page, **metadata)
        if validate:
            self.validate(doc)
        return doc

    def validate(self, doc: str) -> None:
        """Validate document"""

    def _serialize(self, page: PageNode, **metadata) -> str | None:
        """Format-specific seralization method

        Arguments:
            page: Input page
        """


class AltoXML(Serializer):
    """Alto XML serializer.

    This serializer uses a jinja template to produce Alto XML files
    according to version 4.4 of the Alto schema.

    # Features
    - Uses Alto version 4.4.
    - Includes detailed processing metadata in the `<Description>` block.
    - Supports rendering of region locations (printspace and margins).
    To enable this, first make sure that the regions are tagged by
    calling `layout.label_regions(...)` before serialization.
    - Will always produce a file, but the file may be empty.

    # Limitations
    - Two-level segmentation: The Alto schema only supports two-level
    segmentation, i.e. pages with regions and lines. Pages with deeper
    segmentation will be flattened so that only the innermost regions
    are rendered.
    - Only includes text confidence at the page level.

    # Examples
    Example usage with the `Export` pipeline step:
    ```yaml
    - step: Export
      settings:
        dest: alto-ouptut
        format: AltoXML
    ```
    """

    extension = ".xml"

    def __init__(self, template_dir=_TEMPLATES_DIR, template_name="alto"):
        """
        Arguments:
            template_dir: Name of template directory.
            template_name: Name of template file in `template_dir`.
        """
        env = Environment(loader=FileSystemLoader([template_dir, "."]))
        self.template = env.get_template(template_name)
        self.schema = os.path.join(_SCHEMA_DIR, "alto-4-4.xsd")

    def _serialize(self, page: PageNode, **metadata) -> str:
        # Find all nodes that correspond to Alto TextBlock elements and
        # their location (if available). A TextBlock is a region whose
        # children are text lines (and not other regions). If the node's
        # `region_location` attribute is set, it will be rendered in the
        # corresponding Alto group, otherwise it will be rendered in the
        # printspace group.
        text_blocks = defaultdict(list)
        for node in page.traverse():
            if node.is_region() and all(child.text for child in node):
                text_blocks[node.get(REGION_KEY, RegionLocation.PRINTSPACE)].append(node)

        return self.template.render(
            page=page,
            page_confidence=average_text_confidence(page),
            printspace=text_blocks[RegionLocation.PRINTSPACE],
            top_margin=text_blocks[RegionLocation.MARGIN_TOP],
            bottom_margin=text_blocks[RegionLocation.MARGIN_BOTTOM],
            left_margin=text_blocks[RegionLocation.MARGIN_LEFT],
            right_margin=text_blocks[RegionLocation.MARGIN_RIGHT],
            metadata=get_metadata(),
            processing_steps=metadata.pop("processing_steps", []),
            xmlescape=xmlescape,
        )

    def validate(self, doc: str) -> None:
        """Validate `doc` against the current schema

        Arguments:
            doc: Input document

        Raises:
            xmlschema.XMLSchemaValidationError if the document violates
            the current schema.
        """
        xmlschema.validate(doc, self.schema)


class PageXML(Serializer):
    """Page XML serializer

    This serializer uses a jinja template to produce Page XML files
    according to the 2019-07-15 version of the schema.

    # Features
    - Includes line confidence scores.
    - Supports nested segmentation.

    # Limitations
    - Will not create an output file if the page is not serializable,
    for example if it does not contain any regions. (This behaviour
    differs from the Alto serializer, which instead would produce an
    empty file.)

    # Examples
    Example usage with the `Export` pipeline step:
    ```yaml
    - step: Export
      settings:
        dest: page-ouptut
        format: PageXML
    ```
    """

    extension = ".xml"

    def __init__(self, template_dir=_TEMPLATES_DIR, template_name="page"):
        """
        Arguments:
            template_dir: Name of template directory.
            template_name: Name of template file in `template_dir`.
        """
        env = Environment(loader=FileSystemLoader([template_dir, "."]))
        self.template = env.get_template(template_name)
        self.schema = os.path.join(_SCHEMA_DIR, "pagecontent.xsd")

    def _serialize(self, page: PageNode, **metadata):
        if page.is_leaf():
            return None

        return self.template.render(
            page=page,
            TEXT_RESULT_KEY=TEXT_RESULT_KEY,
            metadata=get_metadata(),
            is_text_line=lambda node: node.is_line(),
            xmlescape=xmlescape,
        )

    def validate(self, doc: str) -> None:
        """Validate `doc` against the current schema

        Arguments:
            doc: Input document

        Raises:
            xmlschema.XMLSchemaValidationError if the document violates
            the current schema.
        """
        xmlschema.validate(doc, self.schema)


class Json(Serializer):
    """
    JSON serializer

    This serializer extracts all content from the collection and saves
    it as json. The resulting json file(s) include properties that are
    not supported by Alto or Page XML, such as region confidence scores.

    # Examples
    Example usage with the `Export` pipeline step:
    ```yaml
    - step: Export
      settings:
        dest: json-ouptut
        format: json
        indent: 2
    ```
    """

    extension = ".json"

    def __init__(self, indent=4):
        """
        Arguments:
            indent: The indentation level of the output json file(s).
        """
        self.indent = indent

    def _serialize(self, page: PageNode, **metadata):
        def default(obj):
            return obj.__dict__

        return json.dumps(page.asdict() | metadata, default=default, indent=self.indent, ensure_ascii=False)


class PlainText(Serializer):
    """
    Plain text serializer

    This serializer extracts all text content from the collection and
    saves it as plain text. All other data (metadata, coordinates,
    geometries, confidence scores, and so on) is discarded.

    # Examples
    Example usage with the `Export` pipeline step:
    ```yaml
    - step: Export
      settings:
        dest: text-ouptut
        format: plaintext
    ```
    """

    extension = ".txt"

    def _serialize(self, page: PageNode, **metadata) -> str:
        lines = page.traverse(lambda node: node.is_line())
        return "\n".join(line.text.strip() for line in lines)


def get_metadata() -> dict:
    timestamp = datetime.utcnow().isoformat()

    return {
        "creator": htrflow.meta["Name"],
        "software_name": htrflow.meta["Name"],
        "software_version": htrflow.meta["Version"],
        "application_description": htrflow.meta["Summary"],
        "created": timestamp,
    }


def get_serializer(serializer_name: str, **serializer_args) -> Serializer:
    names = {serializer.__name__.lower(): serializer for serializer in Serializer.__subclasses__()}

    aliases = {
        "alto": AltoXML,  # for backwards compatibility
        "page": PageXML,  # for backwards compatibility
        "txt": PlainText,  # for backwards compatibility
        "text": PlainText,  # for convenience
    }

    serializer = (names | aliases).get(serializer_name.lower(), None)
    if serializer is None:
        raise ValueError(f"Format '{serializer_name}' is not among the supported formats: {', '.join(names)}")

    return serializer(**serializer_args)


def save_collection(collection: Collection, serializer: str | Serializer, dest: str, **metadata):
    """Serialize and save collection

    Arguments:
        collection: Input collection
        serializer: What serializer to use. Takes a Serializer instance
            or the name of the serializer as a string.
        dest: Output directory
    """

    for page in collection:
        page.prune(lambda node: node.is_leaf() and node.depth != page.max_depth())
    collection.relabel()

    if isinstance(serializer, str):
        serializer = get_serializer(serializer)
        logger.info("Using %s serializer with default settings", serializer.__class__.__name__)

    for page in collection:
        doc = serializer.serialize(page, **metadata)
        if doc is None:
            continue
        filename = os.path.join(collection.label, page.label + serializer.extension)
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
