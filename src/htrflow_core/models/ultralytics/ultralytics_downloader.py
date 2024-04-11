from os import PathLike
from pathlib import Path
from typing import Optional

from htrflow_core.models import hf_utils


SUPPORTED_MODEL_TYPES = (".pt", ".yaml")


def load_from_hf(model_id: str, cache_dir: str | PathLike = "./.cache", hf_token: Optional[str] = None) -> Path:
    cache_dir = Path(cache_dir)
    model_path = Path(model_id)

    if model_path.exists() and model_path.suffix not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Please provide model of type: {SUPPORTED_MODEL_TYPES}")
    elif model_path.exists() and model_path.suffix in SUPPORTED_MODEL_TYPES:
        return model_path

    repo_files = hf_utils.list_files_from_repo(model_id, hf_token)

    _ = hf_utils.hf_download_counter(model_id, cache_dir, hf_token, repo_files)

    model_file = hf_utils.find_supported_file(repo_files, SUPPORTED_MODEL_TYPES, model_id, "model")
    return Path(hf_utils.download_file(model_id, model_file, cache_dir, hf_token))
