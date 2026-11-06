"""
Model resolution for kraken models.

HTRflow's `models.download` only knows the Hugging Face Hub, but kraken
models are usually published in the htrmopo repository on Zenodo. This
module accepts the four things that can reasonably appear in a YAML
`model:` field:

    model: null                              # kraken's bundled blla.mlmodel
    model: path/to/my-model.mlmodel          # a local file or directory
    model: 10.xxxx/zenodo.xxxxxxx            # an htrmopo DOI
    model: SomeUser/some-kraken-repo         # a Hugging Face repo

and returns a concrete file path plus a version string for the processing
metadata that ends up in the exported XML.
"""

import logging
import os
import re
from pathlib import Path

from htrflow.models.download import _hf_hub_download_matching_file, get_model_info


logger = logging.getLogger(__name__)

# Weight formats kraken's `load_models` knows how to deserialize, most
# preferred first. Anything else in a model directory (READMEs, metadata) is
# ignored.
MODEL_SUFFIXES = (".safetensors", ".mlmodel")

_DOI = re.compile(r"^10\.\d{4,9}/\S+$")


def resolve_kraken_model(model: str | None, revision: str | None = None) -> tuple[str, str]:
    """
    Resolve `model` to a local kraken weights file.

    Arguments:
        model: A local path, an htrmopo DOI, a Hugging Face repo ID, or None
            to use the segmentation model bundled with kraken.
        revision: Optional revision, only meaningful for Hugging Face repos.

    Returns:
        A `(path, version)` tuple, where `version` identifies the exact
        weights for reproducibility ("local", a DOI, or a commit hash).

    Raises:
        FileNotFoundError: if no usable weights file could be found.
    """
    if model is None:
        return _bundled_blla(), "kraken-bundled"

    # `~` is how people write paths in YAML, but pathlib only expands it on request.
    path = Path(model).expanduser()
    if path.is_file():
        logger.info("Using local kraken model %s", path)
        return str(path), "local"
    if path.is_dir():
        return str(_first_model_file(path)), "local"

    if _DOI.match(model):
        return _from_htrmopo(model)

    # Anything left is treated as a Hugging Face repo ID. Catch the obvious
    # typo first: something that looks like a path the user got wrong should
    # not come back as a complaint about Hugging Face repo naming.
    if _looks_like_a_path(model):
        if path.is_absolute():
            where = f"{path} does not exist"
        else:
            where = f"no such file or directory exists relative to {os.getcwd()}"
        raise FileNotFoundError(f"No kraken model at '{model}'. It looks like a file path, but {where}.")

    return _from_huggingface(model, revision)


def _looks_like_a_path(model: str) -> bool:
    """
    Whether `model` was probably meant as a local path rather than a repo ID.

    Hugging Face repo IDs are `owner/name`, which is annoyingly close to a
    relative path. These three signals mean a path was intended: an explicit
    path prefix, a weights file extension, or more separators than a repo ID
    can have.
    """
    return model.startswith((".", "~", "/")) or model.endswith(MODEL_SUFFIXES) or model.count("/") > 1


def _bundled_blla() -> str:
    """Path to the default baseline segmentation model shipped with kraken."""
    from importlib import resources

    path = resources.files("kraken").joinpath("blla.mlmodel")
    if not path.is_file():
        raise FileNotFoundError(
            "No model given and kraken's bundled blla.mlmodel is not available at "
            f"{path}. Pass an explicit `model` instead."
        )
    logger.info("No model given, using kraken's bundled segmentation model at %s", path)
    return str(path)


def _first_model_file(directory: Path) -> Path:
    """Pick the weights file out of a directory of downloaded model artifacts."""
    for suffix in MODEL_SUFFIXES:
        candidates = sorted(directory.glob(f"*{suffix}"))
        if candidates:
            if len(candidates) > 1:
                logger.warning(
                    "Found %d %s files in %s, using %s",
                    len(candidates),
                    suffix,
                    directory,
                    candidates[0].name,
                )
            return candidates[0]
    raise FileNotFoundError(f"No kraken model file ({' or '.join(MODEL_SUFFIXES)}) found in {directory}.")


def _from_htrmopo(doi: str) -> tuple[str, str]:
    """Download a model from the htrmopo repository, or reuse the cached copy."""
    import htrmopo

    logger.info("Fetching kraken model %s from the htrmopo repository", doi)
    directory = Path(htrmopo.get_model(doi))
    return str(_first_model_file(directory)), doi


def _from_huggingface(repo_id: str, revision: str | None) -> tuple[str, str]:
    """Download the first kraken-loadable file from a Hugging Face repo."""
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

    for suffix in MODEL_SUFFIXES:
        try:
            path = _hf_hub_download_matching_file(repo_id, f"*{suffix}", revision)
        except FileNotFoundError:
            continue
        except (HFValidationError, RepositoryNotFoundError) as e:
            # Don't let Hugging Face's own wording surface here: at this point
            # all we know is that none of the four accepted forms matched.
            raise FileNotFoundError(_unresolvable(repo_id)) from e
        return path, get_model_info(repo_id, revision) or "unknown"

    raise FileNotFoundError(_unresolvable(repo_id))


def wrong_task_message(path: str, model: str | None, wanted: str) -> str:
    """
    Explain that a model file does not perform the task it was asked to.

    kraken raises for this, but its message either dumps the whole model repr
    or says "model list []" after filtering, neither of which tells the user
    which YAML key to fix. Only called on the error path, so re-reading the
    file is free in the normal case.
    """
    try:
        from kraken.models import load_models

        tasks = sorted({task for m in load_models(path) for task in m.model_type})
    except Exception:
        tasks = []

    name = model or path
    if tasks:
        lead = f"'{name}' performs {' and '.join(tasks)}, but this step needs a {wanted} model."
    else:
        lead = f"'{name}' could not be loaded as a {wanted} model."

    return (
        f"{lead} Check which model is passed to which step: segmentation models belong to "
        f"the `Segmentation` step with `model: KrakenSegmentation`, recognition models to "
        f"the `KrakenTextRecognition` step with `model: KrakenRecognition`."
    )


def _unresolvable(model: str) -> str:
    """The message for a `model` that matched none of the accepted forms."""
    return (
        f"Could not resolve kraken model '{model}'. It is not an existing file or "
        f"directory, not an htrmopo DOI (of the form 10.xxxx/zenodo.xxxxxxx), and not a "
        f"Hugging Face repo containing a {' or '.join(MODEL_SUFFIXES)} file. Local "
        f"paths are resolved against the working directory, currently {os.getcwd()}."
    )
