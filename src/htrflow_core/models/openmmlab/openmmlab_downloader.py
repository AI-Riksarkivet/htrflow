# openmmlab_downloader.py
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

from mmengine.config import Config

from htrflow_core.models import hf_utils


SUPPORTED_MODEL_TYPES = (".pth",)
CONFIG_FILE = "config.py"
DICT_FILE = "dictionary.txt"


def load_from_hf(
    model_id: str, config_id: str, cache_dir: str | PathLike = "./.cache", hf_token: Optional[str] = None
) -> Tuple[PathLike, PathLike]:
    cache_dir = Path(cache_dir)

    if config_id is None:
        config_id = model_id

    model_path = Path(model_id)
    config_path = Path(config_id)

    if model_path.exists() and model_path.suffix not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Please provide model of type: {SUPPORTED_MODEL_TYPES}")
    elif model_path.exists() and model_path.suffix in SUPPORTED_MODEL_TYPES:
        if config_path.exists() and config_path.suffix == ".py":
            return model_id, config_id
        elif config_path.exists() and config_path.suffix != ".py":
            raise ValueError(f"Please provide config of type: {CONFIG_FILE}")

    repo_files = hf_utils.list_files_from_repo(model_id, hf_token)

    _ = hf_utils.hf_download_counter(model_id, cache_dir, hf_token, repo_files)
    model_file = hf_utils.find_supported_file(repo_files, SUPPORTED_MODEL_TYPES, model_id, "model")
    model_path = hf_utils.download_file(model_id, model_file, cache_dir, hf_token)

    config_file = hf_utils.find_supported_file(repo_files, [CONFIG_FILE], model_id, "config")
    config_path = hf_utils.download_file(model_id, config_file, cache_dir, hf_token)

    dict_file = hf_utils.find_supported_file(repo_files, [DICT_FILE], model_id, "dictionary", required=False)

    if dict_file:
        dictionary_path = hf_utils.download_file(model_id, dict_file, cache_dir, hf_token)
        configure_dictionary(config_path, dictionary_path)

    return model_path, config_path


def configure_dictionary(config_path: Path, dictionary_path: Path):
    cfg = Config.fromfile(config_path)
    cfg.dictionary["dict_file"] = dictionary_path
    cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_path
    cfg.dump(config_path)
