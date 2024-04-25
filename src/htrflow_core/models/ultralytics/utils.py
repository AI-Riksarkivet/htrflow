from os import PathLike
from pathlib import Path

from htrflow_core.models import hf_utils


ULTRALYTICS_SUPPORTED_MODEL_TYPES = (".pt", ".yaml")


def ultralytics_from_hf(
    model_id: str,
    cache_dir: str | PathLike = "./.cache",
    supported_model_types=ULTRALYTICS_SUPPORTED_MODEL_TYPES,
) -> str:
    existing_model = validate_and_return_existing_model(model_id, supported_model_types)
    if existing_model:
        return existing_model

    repo_files = hf_utils.count_and_list_files_in_repo(model_id, cache_dir)
    return hf_utils.download_file_from_hf(model_id, cache_dir, repo_files)


def validate_and_return_existing_model(model_id: str, supported_model_types: tuple) -> Path:
    model_path = Path(model_id)
    if model_path.exists():
        if model_path.suffix in supported_model_types:
            return model_path
        raise ValueError(f"Please provide model of type: {supported_model_types}")
    return None
