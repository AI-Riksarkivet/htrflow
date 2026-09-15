"""
Model resolution for kraken models.

kraken distributes its models through the htrmopo/Zenodo repository rather
than the Hugging Face hub, so `models.download` only covers part of the
problem. This module accepts the four things that can reasonably appear in a
YAML `model:` field:

    model: null                              # kraken's bundled blla.mlmodel
    model: path/to/my-model.mlmodel          # a local file or directory
    model: 10.5281/zenodo.21788409           # an htrmopo DOI
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

    path = Path(model)
    if path.is_file():
        logger.info("Using local kraken model %s", path)
        return str(path), "local"
    if path.is_dir():
        return str(_first_model_file(path)), "local"

    if _DOI.match(model):
        return _from_htrmopo(model)

    return _from_huggingface(model, revision)


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
                    len(candidates), suffix, directory, candidates[0].name,
                )
            return candidates[0]
    raise FileNotFoundError(
        f"No kraken model file ({' or '.join(MODEL_SUFFIXES)}) found in {directory}."
    )


def _from_htrmopo(doi: str) -> tuple[str, str]:
    """Download a model from the htrmopo repository, or reuse the cached copy."""
    import htrmopo

    logger.info("Fetching kraken model %s from the htrmopo repository", doi)
    directory = Path(htrmopo.get_model(doi))
    return str(_first_model_file(directory)), doi


def _from_huggingface(repo_id: str, revision: str | None) -> tuple[str, str]:
    """Download the first kraken-loadable file from a Hugging Face repo."""
    for suffix in MODEL_SUFFIXES:
        try:
            path = _hf_hub_download_matching_file(repo_id, f"*{suffix}", revision)
        except FileNotFoundError:
            continue
        return path, get_model_info(repo_id, revision) or "unknown"

    raise FileNotFoundError(
        f"Could not resolve kraken model '{repo_id}'. It is not a local path, not a "
        f"DOI, and no {' or '.join(MODEL_SUFFIXES)} file was found in the Hugging "
        f"Face repo of that name. Local paths are checked against the working "
        f"directory, which is currently {os.getcwd()}."
    )
