"""
Module for pretty-printing progress.
"""

import atexit
from collections import defaultdict
from typing import TYPE_CHECKING

from rich.progress import Progress, SpinnerColumn, TextColumn

from htrflow.document import Document


if TYPE_CHECKING:
    from htrflow.pipeline.steps import PipelineStep


# A mapping from `document` to task ID
_tasks = {}
_exports = defaultdict(list)
_steps = defaultdict(list)

# The progress singleton
_progress = Progress(
    TextColumn("{task.fields[filename]}:"),
    SpinnerColumn(),
    TextColumn("Running '{task.fields[status]}'"),
)


_progress.start()
atexit.register(_progress.stop)


def update(document: Document, *args, **kwargs):
    """
    Update progress for the given `document`.
    """
    if document not in _tasks:
        _register(document)
    _progress.update(_tasks[document], *args, **kwargs)


def step(document: Document, step: "PipelineStep"):
    """
    Register processing step for the given `document`.
    """
    update(document, status=str(step))
    _steps[document].append(step)


def get_steps(document: Document) -> list["PipelineStep"]:
    """
    Return a list of all steps registered for this document.
    """
    return _steps[document]


def done(document: Document):
    """
    Register `document` as done

    Disables its progress bar and prints a summary to the terminal.
    """
    exports = _exports[document]
    if exports:
        msg = "Results exported to " + ", ".join(exports)
    else:
        msg = "No results exported."

    _progress.print(f"{document.image_name}: Done! [dim]{msg}[/dim]")
    update(document, visible=False)


def register_export(document: Document, path: str):
    """
    Register an export path for `document`

    The path will be printed when the document is done.
    """
    _exports[document].append(path)


def _register(document: Document):
    """
    Register a task for the given `document`.
    """
    _tasks[document] = _progress.add_task("task", filename=document.image_name, status="")
