# hf_utils.py
import logging
from os import PathLike
from typing import List, Optional

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError


META_MODEL_TYPE = "model"
CONFIG_JSON = "config.json"


def download_file(
    repo_id: str, filename: str, cache_dir: PathLike, hf_token: Optional[str], repo_type: str = META_MODEL_TYPE
) -> PathLike:
    """Download a file from the Hugging Face Hub.

    Args:
        repo_id: Repository ID.
        filename: Name of the file to download.
        cache_dir: Directory to cache the downloaded file.
        hf_token: Optional Hugging Face authentication token.
        repo_type: Type of repository, defaults to model.

    Returns:
        Path to the downloaded file.
    """
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type, cache_dir=cache_dir, token=hf_token
        )
    except Exception as e:
        logging.error(f"Failed to download {filename} from {repo_id}: {e}")
        raise


def list_files_from_repo(repo_id: str, token: Optional[str]) -> List[str]:
    """List files available in a Hugging Face Hub repository.

    Args:
        repo_id: Repository ID.
        token: Optional Hugging Face authentication token.

    Returns:
        List of file names in the repository.

    Raises:
        RepositoryNotFoundError: If the repository is not found.
    """
    try:
        return list_repo_files(repo_id=repo_id, repo_type=META_MODEL_TYPE, token=token)
    except RepositoryNotFoundError as e:
        logging.error(f"Could not download files for {repo_id}: {str(e)}")
        raise ValueError(f"Repository {repo_id} not found.") from e


def find_supported_file(files: List[str], extensions: List[str], repo_id: str, file_type: str, required: bool = True):
    file = next((f for f in files if any(f.endswith(ext) for ext in extensions)), None)
    if required and not file:
        raise ValueError(f"No {file_type} file of supported type: {extensions} found in repository {repo_id}.")
    return file


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
