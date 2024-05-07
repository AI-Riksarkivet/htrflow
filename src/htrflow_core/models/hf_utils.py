import logging
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError


# TODO: add pytest

logger = logging.getLogger(__name__)


class HFBaseDownloader:
    META_MODEL_TYPE = "model"
    META_CONFIG_TYPE = "config"
    META_DICT_TYPE = "dictionary"

    CONFIG_JSON = "config.json"
    PY_EXTENSION = ".py"

    ULTRALYTICS_SUPPORTED_MODEL_TYPES = (".pt",)  # ".yaml"

    def list_files_from_repo(self, repo_id: str) -> List[str]:
        """List files available in a Hugging Face Hub repository."""
        try:
            repo_files = list_repo_files(repo_id=repo_id, repo_type=self.META_MODEL_TYPE, token=HF_CONFIG["token"])
            _ = self._hf_download_counter(repo_id, repo_files)
            return repo_files
        except RepositoryNotFoundError as e:
            logging.error(f"Could not download files for {repo_id}: {str(e)}")
            raise

    def _hf_download_counter(self, model_id, repo_files):
        if self.CONFIG_JSON in repo_files:
            return hf_hub_download(
                repo_id=model_id, filename=self.CONFIG_JSON, repo_type=self.META_MODEL_TYPE, **HF_CONFIG
            )

    def wrapper_hf_hub_download(self, repo_id: str, filename: str, repo_type: str = META_MODEL_TYPE) -> str:
        """Download a file from the Hugging Face Hub."""
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                cache_dir=HF_CONFIG["cache_dir"],
                token=HF_CONFIG["token"],
            )
        except Exception as e:
            logging.error(f"Failed to download {filename} from {repo_id}: {e}")
            raise

    def find_supported_file(
        self, repo_files: List[str], supported_types: List[str], repo_id: str, file_type: str, required: bool = True
    ):
        file = next((f for f in repo_files if any(f.endswith(ext) for ext in supported_types)), None)
        if required and not file:
            raise ValueError(
                f"No {file_type} file of supported type: {supported_types} found in repository {repo_id}."
            )
        return file

    def _download_file_from_hf(self, repo_id, supported_file_extension, file_type, repo_files):
        file_name = self.find_supported_file(repo_files, supported_file_extension, repo_id, file_type)
        return self.wrapper_hf_hub_download(repo_id, file_name)


class MMLabsDownloader(HFBaseDownloader):
    MMLABS_CONFIG_FILE = "config.py"
    MMLABS_DICT_FILE = "dictionary.txt"
    MMLABS_SUPPORTED_MODEL_TYPES = (".pth",)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        config_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Download and load config and model from Openmmlabs using the HuggingFace Hub."""

        downloader = cls()
        config_id = config_id or model_id
        existing_model = downloader._mmlab_try_load_from_local_files(model_id, config_id)

        if existing_model:
            logging.info(f"Loaded existing model from '{existing_model}'")
            return existing_model

        repo_files = downloader.list_files_from_repo(model_id)
        model_path = downloader._download_file_from_hf(
            model_id, cls.MMLABS_SUPPORTED_MODEL_TYPES, cls.META_MODEL_TYPE, repo_files
        )
        config_path = downloader._download_file_from_hf(
            model_id, [cls.MMLABS_CONFIG_FILE], cls.META_CONFIG_TYPE, repo_files
        )

        dict_file = downloader.find_supported_file(
            repo_files, [cls.MMLABS_DICT_FILE], model_id, cls.META_DICT_TYPE, required=False
        )
        if dict_file:
            dictionary_path = downloader.wrapper_hf_hub_download(model_id, dict_file)
            downloader._fix_mmlab_dict_file(config_path, dictionary_path)

        logging.info(f"Downloaded model '{model_id}' from HF and loaded it from folder: '{HF_CONFIG['cache_dir']}'")

        return model_path, config_path

    def _mmlab_try_load_from_local_files(self, model_id, config_id) -> Optional[Tuple[str, str]]:
        model_path = Path(model_id)
        config_path = Path(config_id)
        if model_path.exists() and model_path.suffix in self.MMLABS_SUPPORTED_MODEL_TYPES:
            if config_path.exists() and config_path.suffix == self.PY_EXTENSION:
                return str(model_id), str(config_id)
            elif config_path.exists() and config_path.suffix != self.PY_EXTENSION:
                raise ValueError(f"Please provide config of type: {self.MMLABS_CONFIG_FILE}")
        return None

    def _fix_mmlab_dict_file(self, config_path: str, dictionary_path: str) -> None:
        from mmengine.config import Config

        cfg = Config.fromfile(config_path)
        cfg.dictionary["dict_file"] = dictionary_path
        cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_path
        cfg.dump(config_path)


class UltralyticsDownloader(HFBaseDownloader):
    ULTRALYTICS_SUPPORTED_MODEL_TYPES = (".pt",)

    @classmethod
    def from_pretrained(cls, model_id: str) -> str:
        """Download and load model from Ultralytics using the HuggingFace Hub."""
        downloader = cls()
        existing_model = downloader._ultralytics_try_load_from_local_files(model_id)
        if existing_model:
            logging.info(f"Loaded existing model from '{existing_model}'")
            return existing_model

        repo_files = downloader.list_files_from_repo(model_id)
        cache_model_path = downloader._download_file_from_hf(
            model_id, cls.ULTRALYTICS_SUPPORTED_MODEL_TYPES, cls.META_MODEL_TYPE, repo_files
        )
        logging.info(f"Downloaded model '{model_id}' from HF and loaded it from folder: '{HF_CONFIG['cache_dir']}'")
        return cache_model_path

    @classmethod
    def _ultralytics_try_load_from_local_files(cls, model_id: str) -> Optional[str]:
        """Check for an existing local file for the model."""
        model_path = Path(model_id)
        if model_path.exists() and model_path.suffix in cls.ULTRALYTICS_SUPPORTED_MODEL_TYPES:
            return str(model_path)
        return None


# Configuration settings for communications with the huggingface hub
HF_CONFIG = {
    "cache_dir": ".cache",
    "local_files_only": False,
    "token": False,
}
