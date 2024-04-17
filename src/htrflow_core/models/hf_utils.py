import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError


META_MODEL_TYPE = "model"
CONFIG_JSON = "config.json"
PY_EXTENSION = ".py"


def list_files_from_repo(repo_id: str, cache_dir: str, hf_token: Optional[str]) -> List[str]:
    """List files available in a Hugging Face Hub repository."""
    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type=META_MODEL_TYPE, token=hf_token)
        _ = hf_download_counter(repo_id, cache_dir, hf_token, repo_files)
        return repo_files
    except RepositoryNotFoundError as e:
        logging.error(f"Could not download files for {repo_id}: {str(e)}")
        raise ValueError(f"Repository {repo_id} not found.") from e


def hf_download_counter(model_id, cache_dir, hf_token, repo_files):
    if CONFIG_JSON in repo_files:
        _ = hf_hub_download(
            repo_id=model_id,
            filename=CONFIG_JSON,
            repo_type=META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )
        return _


def wrapper_hf_hub_download(
    repo_id: str, filename: str, cache_dir: str, hf_token: Optional[str], repo_type: str = META_MODEL_TYPE
) -> str:
    """Download a file from the Hugging Face Hub."""
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type, cache_dir=cache_dir, token=hf_token
        )
    except Exception as e:
        logging.error(f"Failed to download {filename} from {repo_id}: {e}")
        raise


def find_supported_file(
    repo_files: List[str], supported_types: List[str], repo_id: str, file_type: str, required: bool = True
):
    file = next((f for f in repo_files if any(f.endswith(ext) for ext in supported_types)), None)
    if required and not file:
        raise ValueError(f"No {file_type} file of supported type: {supported_types} found in repository {repo_id}.")
    return file


def download_file_from_hf(repo_id, supported_file_extension, file_type, repo_files, cache_dir, hf_token):
    file_name = find_supported_file(repo_files, supported_file_extension, repo_id, file_type)
    return wrapper_hf_hub_download(repo_id, file_name, cache_dir, hf_token)


MMLABS_CONFIG_FILE = "config.py"
MMLABS_DICT_FILE = "dictionary.txt"
MMLABS_SUPPORTED_MODEL_TYPES = (".pth",)


def mmlabs_from_hf(
    model_id: str,
    config_id: str,
    cache_dir: str = "./.cache",
    hf_token: Optional[str] = None,
) -> Tuple[str, str]:
    existing_model = mmlab_try_load_from_local_files(
        model_id, config_id, MMLABS_SUPPORTED_MODEL_TYPES, MMLABS_CONFIG_FILE
    )
    if existing_model:
        return existing_model

    repo_id = model_id

    repo_files = list_files_from_repo(repo_id, cache_dir, hf_token)

    model_path = download_file_from_hf(
        repo_id, MMLABS_SUPPORTED_MODEL_TYPES, META_MODEL_TYPE, repo_files, cache_dir, hf_token
    )

    config_path = download_file_from_hf(repo_id, [MMLABS_CONFIG_FILE], "config", repo_files, cache_dir, hf_token)

    dict_file = find_supported_file(repo_files, [MMLABS_DICT_FILE], repo_id, "dictionary", required=False)

    if dict_file:
        dictionary_path = wrapper_hf_hub_download(model_id, dict_file, cache_dir, hf_token)
        fix_mmlab_dict_file(config_path, dictionary_path)

    return model_path, config_path


def mmlab_try_load_from_local_files(model_id, config_id, supported_model_types, config_file):
    config_id = config_id or model_id

    model_path = Path(model_id)
    config_path = Path(config_id)

    if model_path.exists() and model_path.suffix not in supported_model_types:
        raise ValueError(f"Please provide model of type: {supported_model_types}")
    elif model_path.exists() and model_path.suffix in supported_model_types:
        if config_path.exists() and config_path.suffix == PY_EXTENSION:
            return model_id, config_id
        elif config_path.exists() and config_path.suffix != PY_EXTENSION:
            raise ValueError(f"Please provide config of type: {config_file}")
    return None


def fix_mmlab_dict_file(config_path: str, dictionary_path: str) -> None:
    from mmengine.config import Config  # noqa: F811

    cfg = Config.fromfile(config_path)
    cfg.dictionary["dict_file"] = dictionary_path
    cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_path
    cfg.dump(config_path)


ULTRALYTICS_SUPPORTED_MODEL_TYPES = (".pt",)  # ".yaml"


def ultralytics_from_hf(
    model_id: str,
    cache_dir: str | PathLike = "./.cache",
    hf_token: Optional[str] = None,
) -> str:
    existing_model = ultralytics_try_load_from_local_files(model_id)
    if existing_model:
        return existing_model

    repo_id = model_id

    repo_files = list_files_from_repo(repo_id, cache_dir, hf_token)
    return download_file_from_hf(
        repo_id, ULTRALYTICS_SUPPORTED_MODEL_TYPES, META_MODEL_TYPE, repo_files, cache_dir, hf_token
    )


def ultralytics_try_load_from_local_files(model_id: str) -> Path:
    model_path = Path(model_id)
    if model_path.exists():
        if model_path.suffix in ULTRALYTICS_SUPPORTED_MODEL_TYPES:
            return model_path
        raise ValueError(f"Please provide model of type: {ULTRALYTICS_SUPPORTED_MODEL_TYPES}")
    return None
