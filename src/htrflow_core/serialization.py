from __future__ import annotations

import datetime
import warnings
from typing import TYPE_CHECKING, Callable, Tuple

import xmlschema
from jinja2 import Environment, FileSystemLoader

import htrflow_core


if TYPE_CHECKING:
    from htrflow_core.volume import Volume


# Path to templates
_TEMPLATES_DIR = "src/htrflow_core/templates"

# Default metadata
_METADATA = {
    "creator": f"{htrflow_core.__author__}",
    "software_name": f"{htrflow_core.__package_name__}",
    "software_version": f"{htrflow_core.__version__}",
    "application_description": f"{htrflow_core.__desc__}",
}

# Mapping of format name -> serializer function
_SERIALIZERS = {
    "alto": lambda volume: serialize_xml(volume, "alto"),
    "page": lambda volume: serialize_xml(volume, "page"),
    "txt": lambda volume: serialize_txt(volume),
}

# Mapping of (XML) format name -> XML schema
_SCHEMAS = {
    "alto": "http://www.loc.gov/standards/alto/v4/alto-4-4.xsd",
    "page": "https://www.primaresearch.org/schema/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
}

def supported_formats():
    """The supported formats"""
    return _SERIALIZERS.keys()


def get_serializer(format_: str) -> Callable[[Volume], list[Tuple[str, str]]]:
    """Get the serializer function associated with `format_`.

    See `serialization.supported_formats()` for supported formats.

    Arguments:
        format_: The output format

    Returns:
        A function that takes a volume and returns a list of (text, filename)
        tuples, where `text` is a serialized page according to the specified
        format, and `filename` is a suggested filename with appropriate file
        extension.

    Raises:
        ValueError: If the format is not supported
    """
    if format_ not in _SERIALIZERS:
        raise ValueError(f"The specified format {format_} is not among the supported formats: {supported_formats()}")
    return _SERIALIZERS[format_]


def serialize_xml(volume: Volume, template: str) -> list[Tuple[str, str]]:
    """Serialize `volume` according to `template`

    Arguments:
        volume: The input volume
        template: Path to jinja template. Will search in `_TEMPLATES_DIR` and
            the current working directory.

    Returns:
        A list of (document, filename) tuples
    """

    # Prepare metadata (added to all output files)
    timestamp = datetime.datetime.utcnow().isoformat()
    metadata = _METADATA | {
        "processing_steps": [
            {"description": "step description", "settings": "step settings"}
        ],  # TODO: Document processing steps
        "created": timestamp,
        "last_change": timestamp,
    }

    # Prepare the content of each file
    env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, "."]))
    tmpl = env.get_template(template)
    docs = []
    for page in volume:
        # `blocks` are the parents of the lines (i.e. nodes that contain text)
        blocks = list({node.parent for node in page.lines() if node.parent})
        blocks.sort(key=lambda x: x.id_)
        doc = tmpl.render(page=page, blocks=blocks, metadata=metadata)
        filename = page.image_name + ".xml"
        docs.append((doc, filename))

    # Validate the XML strings against the schema
    schema = _SCHEMAS[template]
    xsd = xmlschema.XMLSchema(schema)
    for doc, filename in docs:
        for error in xsd.iter_errors(doc):
            warnings.warn(
                f"Failed to validate {filename} against {schema}: {error.reason}."
            )

    return docs


def serialize_txt(volume: Volume) -> list[tuple[str, str]]:
    """Seralize volume as plain text

    Args:
        volume: The input volume

    Returns:
        A list of (text, filename) tuples
    """
    docs = []
    for page in volume:
        doc = "\n".join(line.text.top_candidate() for line in page.lines())
        filename = page.image_name + ".txt"
        docs.append((doc, filename))
    return docs
