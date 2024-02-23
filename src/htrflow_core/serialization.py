from __future__ import annotations
import datetime
from typing import TYPE_CHECKING, List, Tuple

from jinja2 import Environment, FileSystemLoader


if TYPE_CHECKING:
    from htrflow_core.volume import Volume


_TEMPLATES_DIR = 'src/htrflow_core/templates'
_METADATA = {
    'creator': 'Riksarkivets AI-labb',
    'software_name': 'HTRFLOW',
    'software_version': f'{__version__}',
    'application_description': '...'
}

_FORMATS = {
    'alto': lambda volume: serialize_xml(volume, 'alto'),
    'page': lambda volume: serialize_xml(volume, 'page'),
    'txt': lambda volume: serialize_txt(volume),
}


def output_formats():
    """The supported formats"""
    return _FORMATS.keys()


def get_serializer(format_):
    return _FORMATS[format_]


def serialize_xml(volume : Volume, template: str) -> List[Tuple[str, str]]:
    """Serialize volume according to `template`

    Arguments:
        volume: The input volume
        template: Path to jinja template. Will search in `_TEMPLATES_DIR` and current working directory.

    Returns:
        A list of (document, filename) tuples
    """

    timestamp = datetime.datetime.utcnow().isoformat()

    metadata = _METADATA | {
        'processing_steps': [{'description': 'step description', 'settings': 'step settings'}],   # TODO: Document processing steps
        'created': timestamp,
        'last_change': timestamp,
    }

    env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, '.']))
    tmpl = env.get_template(template)

    docs = []
    for page in volume:
        # `blocks` are the parents of the lines (i.e. nodes that contain text)
        blocks = list({node.parent for node in page.lines() if node.parent})
        blocks.sort(key=lambda x: x.id_)
        doc = tmpl.render(page=page, blocks=blocks, metadata=metadata)
        filename = page.image_name + '.xml'
        docs.append((doc, filename))
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
        doc = '\n'.join(line.text.top_candidate() for line in page.lines())
        filename = page.image_name + '.txt'
        docs.append((doc, filename))
    return docs
