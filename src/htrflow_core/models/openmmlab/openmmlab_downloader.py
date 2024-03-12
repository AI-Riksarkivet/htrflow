from pathlib import Path
from typing import Optional, Tuple, Union

from huggingface_hub import hf_hub_download, list_repo_files
from mmengine.config import Config

from htrflow_core.models.hf_downloader import HFDownloader


class OpenmmlabDownloader(HFDownloader):
    """
    A subclass of HFDownloader tailored for downloading and configuring OpenMMLab models from the Hugging Face Hub.
    """

    SUPPORTED_MODEL_TYPES = (".pth",)
    CONFIG_FILE = "config.py"
    DICT_FILE = "dictionary.txt"

    @classmethod
    def _download_from_hub(
        cls, hf_model_id: str, cache_dir: Union[str, Path], hf_token: Optional[str]
    ) -> Tuple[str, str]:
        cache_dir = Path(cache_dir)
        repo_files = list_repo_files(repo_id=hf_model_id, repo_type=cls.META_MODEL_TYPE, token=hf_token)

        if cls.CONFIG_JSON in repo_files:
            _ = hf_hub_download(
                repo_id=hf_model_id,
                filename=cls.CONFIG_JSON,
                repo_type=cls.META_MODEL_TYPE,
                cache_dir=cache_dir,
                token=hf_token,
            )

        model_file = cls._find_supported_file(repo_files, cls.SUPPORTED_MODEL_TYPES, hf_model_id, "model")
        config_file = cls._find_supported_file(repo_files, [cls.CONFIG_FILE], hf_model_id, "config")

        model_path = hf_hub_download(
            repo_id=hf_model_id,
            filename=model_file,
            repo_type=cls.META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )
        config_path = hf_hub_download(
            repo_id=hf_model_id,
            filename=config_file,
            repo_type=cls.META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )

        dict_file = cls._find_supported_file(repo_files, [cls.DICT_FILE], hf_model_id, "dictionary", required=False)
        if dict_file:
            dictionary_path = hf_hub_download(
                repo_id=hf_model_id,
                filename=dict_file,
                repo_type=cls.META_MODEL_TYPE,
                cache_dir=cache_dir,
                token=hf_token,
            )
            cls._configure_dictionary(config_path, dictionary_path)

        return config_path, model_path

    @staticmethod
    def _find_supported_file(files, extensions, repo_id, file_type, required=True):
        file = next((f for f in files if any(f.endswith(ext) for ext in extensions)), None)
        if required and not file:
            raise ValueError(f"No {file_type} file of supported type: {extensions} found in repository {repo_id}.")
        return file

    @staticmethod
    def _configure_dictionary(config_path, dictionary_path):
        cfg = Config.fromfile(config_path)
        cfg.dictionary["dict_file"] = dictionary_path
        cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_path
        cfg.dump(config_path)
