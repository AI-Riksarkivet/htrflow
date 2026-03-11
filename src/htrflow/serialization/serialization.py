"""
Serialization module

This module contains functions for exporting a `Collection` or `PageNode`
instance to different formats.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from importlib import metadata

import xmlschema
from jinja2 import Environment, FileSystemLoader

from htrflow.document import Document, Region
from htrflow.progress import get_steps
from htrflow.utils.geometry import Polygon


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

    def serialize(self, document: Document, validate: bool = False) -> str | None:
        """Serialize document

        Arguments:
            document: Input document
            validate: If True, the generated document is passed through
                valiadation before return.

        Returns:
            A string if the serialization was succesful, else None
        """
        doc = self._serialize(document)
        if validate:
            self.validate(doc)
        return doc

    def __str__(self):
        return self.__class__.__name__

    def validate(self, doc: str) -> None:
        """Validate document"""

    def _serialize(self, document: Document) -> str | None:
        """Format-specific seralization method

        Arguments:
            document: Input document
        """


class AltoXML(Serializer):
    """Alto XML serializer.

    This serializer uses a jinja template to produce Alto XML files
    according to version 4.4 of the Alto schema.

    # Features
    - Uses Alto version 4.4.
    - Includes detailed processing metadata in the `<Description>` block.
    - Will always produce a file, but the file may be empty.

    # Limitations
    - Two-level segmentation: The Alto schema only supports two-level
    segmentation, i.e. pages with regions and lines.
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

    def __init__(self, template_dir=_TEMPLATES_DIR, template_name="alto-4-4"):
        """
        Arguments:
            template_dir: Name of template directory.
            template_name: Name of template file in `template_dir`.
        """
        env = Environment(loader=FileSystemLoader([template_dir, "."]))
        self.template = env.get_template(template_name)
        self.schema = os.path.join(_SCHEMA_DIR, template_name + ".xsd")

    def _serialize(self, document: Document) -> str:
        return self.template.render(
            document=document, metadata=metadata.metadata("htrflow"), timestamp=timestamp(), steps=get_steps(document)
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

    def __init__(self, template_dir=_TEMPLATES_DIR, template_name="page2019"):
        """
        Arguments:
            template_dir: Name of template directory.
            template_name: Name of template file in `template_dir`.
        """
        env = Environment(loader=FileSystemLoader([template_dir, "."]))
        self.template = env.get_template(template_name)
        self.schema = os.path.join(_SCHEMA_DIR, template_name + ".xsd")

    def _serialize(self, document: Document):
        return self.template.render(
            document=document,
            metadata=metadata.metadata("htrflow"),
            timestamp=timestamp(),
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

    def __str__(self):
        return f"PageXML(template={self.template.name})"


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

    def __init__(self, **kwargs):
        """
        Arguments:
            **kwargs: key word arguments forwarded to json.dumps()
        """
        self.kwargs = kwargs

    def _serialize(self, document: Document):
        def default(node):
            if isinstance(node, Polygon):
                return str(node)
            attributes = {key: val for key, val in node.__dict__.items() if val}
            return attributes

        return json.dumps(document, default=default, ensure_ascii=False, **self.kwargs)


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

    def _serialize(self, document: Document) -> str:
        return get_text(document)


def get_text(region: Region):
    if not region.transcription:
        return "\n".join(map(get_text, region.regions)) + "\n"
    return max(region.transcription, key=lambda t: t.confidence).text


def timestamp():
    return datetime.now(timezone.utc).isoformat()


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
