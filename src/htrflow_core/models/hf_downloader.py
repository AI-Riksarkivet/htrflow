import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from huggingface_hub.utils import RepositoryNotFoundError


class HFDownloader:
    SUPPORTED_MODEL_TYPES = ()
    CONFIG_JSON = "config.json"
    META_MODEL_TYPE = "model"
    CONFIG_FILE = None
    DICT_FILE = None

    @classmethod
    def from_pretrained(
        cls, model: Union[str, Path], cache_dir: Union[str, Path] = "./.cache", hf_token: Optional[str] = None
    ) -> Union[Path, Tuple[Path, Path], None]:
        """Downloads the model file from Hugging Face Hub or loads it from a local path.

        Args:
            model: The model ID on Hugging Face Hub or a local path to the model.
            cache_dir: The directory where model files should be cached.
            hf_token: An optional Hugging Face authentication token.

        Returns:
            The path to the downloaded files in cache folder.
        """
        model_path = Path(model)
        try:
            if model_path.exists() and model_path.suffix in cls.SUPPORTED_MODEL_TYPES:
                return model_path
            else:
                try:
                    return cls._download_from_hub(model, cache_dir, hf_token)
                except RepositoryNotFoundError as e:
                    logging.error(f"Could not download files for {model}: {str(e)}")
                    return None
        except Exception as e:
            raise FileNotFoundError(f"Model file or Hugging Face Hub model {model} not found.") from e

    @classmethod
    def _download_from_hub(
        cls, hf_model_id: str, cache_dir: Union[str, Path], hf_token: Optional[str]
    ) -> Union[Path, Tuple[Path, ...]]:
        raise NotImplementedError("This method should be implemented by subclasses.")
