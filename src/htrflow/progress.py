"""
Module for pretty-printing progress.
"""

import atexit
from collections import defaultdict

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn

from htrflow.document import Document


# A mapping from `document` to task ID
_tasks = {}
_exports = defaultdict(list)

# The progress singleton
_progress = Progress(
    TextColumn("{task.fields[filename]}:"),
    SpinnerColumn(),
    TextColumn("Running '{task.fields[status]}'"),
)


_quiet = True


def enable():
    """
    Enable progress logging

    With progress logging enabled, HTRflow pretty-prints progress
    updates to the terminal for a human to watch while they wait.
    """
    global _quiet
    _progress.start()
    _quiet = False
    atexit.register(_progress.stop)


def update(document: Document, *args, **kwargs):
    """
    Update progress for the given `document`.
    """
    if document not in _tasks:
        _register(document)
    _progress.update(_tasks[document], *args, **kwargs)


def done(document: Document):
    """
    Register `document` as done

    Disables its progress bar and prints a summary to the terminal.
    """
    if _quiet:
        return

    exports = _exports[document]
    if exports:
        msg = "Results exported to " + ", ".join(exports)
    else:
        msg = "No results exported."
    print(f"{document.image_name}: Done! [dim]{msg}[/dim]")
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
