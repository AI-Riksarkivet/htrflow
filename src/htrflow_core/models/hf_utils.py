import logging

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError


REPO_TYPE = "model"
CACHE_DIR = "./cache"


def download_from_hub(hf_model_id, hf_token=None, cache_dir: str = CACHE_DIR):
    """
    Downloads a model from huggingface hub

    Args:
        hf_model_id (str): huggingface model id to be downloaded from
        hf_token (str): huggingface read token

    Returns:
        model_path (str): path to downloaded model
    """

    repo_files = list_repo_files(repo_id=hf_model_id, repo_type=REPO_TYPE, token=hf_token)

    config_file = "config.json"
    if config_file in repo_files:
        _ = hf_hub_download(
            repo_id=hf_model_id,
            filename=config_file,
            repo_type=REPO_TYPE,
            token=hf_token,
        )

    # download model file
    model_file = [f for f in repo_files if f.endswith(".pt")][0]
    file = hf_hub_download(
        repo_id=hf_model_id,
        filename=model_file,
        repo_type=REPO_TYPE,
        token=hf_token,
    )
    return file


def _download_config_and_model_file(cls, repo_id, cache_dir):
    try:
        model_file = hf_hub_download(
            repo_id=repo_id,
            repo_type=cls.REPO_TYPE,
            filename=cls.MODEL_FILE,
            library_name=__package__,
            cache_dir=cache_dir,
        )
        config_py = hf_hub_download(
            repo_id=repo_id,
            repo_type=cls.REPO_TYPE,
            filename=cls.CONFIG_FILE,
            library_name=__package__,
            cache_dir=cache_dir,
        )
        return model_file, config_py

    except RepositoryNotFoundError as e:
        logging.error(f"Could not download files for {repo_id}: {str(e)}")
        return None, None


@classmethod
def _download_dictonary_file(cls, repo_id, cache_dir):
    try:
        dictionary_file = hf_hub_download(
            repo_id=repo_id,
            repo_type=cls.REPO_TYPE,
            filename=cls.DICT_FILE,
            library_name=__package__,
            cache_dir=cache_dir,
        )

        return dictionary_file

    except RepositoryNotFoundError as e:
        logging.error(f"Could not download files for {repo_id}: {str(e)}")
        return None
