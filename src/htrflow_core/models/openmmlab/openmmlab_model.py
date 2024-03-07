import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError
from mmengine.config import Config


class OpenmmlabModel:
    SUPPORTED_MODEL_TYPES = (".pth",)
    CONFIG_JSON = "config.json"
    META_MODEL_TYPE = "model"
    CONFIG_FILE = "config.py"
    DICT_FILE = "dictionary.txt"

    @classmethod
    def from_pretrained(
        cls, model: str | Path, cache_dir: str | Path = "./.cache", hf_token: Optional[str] = None
    ) -> Path:
        """Downloads the model file from Hugging Face Hub or loads it from a local path.

        Args:
            model: The model ID on Hugging Face Hub or a local path to the model.
            cache_dir: The directory where model files should be cached.
            hf_token: An optional Hugging Face authentication token.

        Returns:
            The path to the downloaded model file.
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
    def _download_from_hub(cls, hf_model_id, cache_dir, hf_token):
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

        model_file = next((f for f in repo_files if f.endswith(cls.SUPPORTED_MODEL_TYPES)), None)

        if not model_file:
            raise ValueError(
                f"No model file of supported type: {cls.SUPPORTED_MODEL_TYPES} found in repository {hf_model_id}."
            )

        model_pth = hf_hub_download(
            repo_id=hf_model_id,
            filename=model_file,
            repo_type=cls.META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )

        config_file = next((f for f in repo_files if f.endswith(cls.CONFIG_FILE)), None)

        if not config_file:
            raise ValueError(f"No config file of supported type: {cls.CONFIG_FILE} found in repository {hf_model_id}.")

        config_py = hf_hub_download(
            repo_id=hf_model_id,
            filename=config_file,
            repo_type=cls.META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )

        dict_file = next((f for f in repo_files if f.endswith(cls.DICT_FILE)), None)

        if dict_file:
            dictionary_file = hf_hub_download(
                repo_id=hf_model_id,
                filename=cls.DICT_FILE,
                repo_type=cls.META_MODEL_TYPE,
                cache_dir=cache_dir,
            )

            cfg = Config.fromfile(config_py)
            cfg.dictionary["dict_file"] = dictionary_file
            cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_file

            cfg.dump(config_py)

        return config_py, model_pth


if __name__ == "__main__":
    # import cv2

    config_py, model_pth = OpenmmlabModel.from_pretrained(model="Riksarkivet/rtmdet_lines")

    from mmdet.apis import DetInferencer

    model = DetInferencer(config_py, model_pth, device="cuda")

    # from mmocr.apis import TextRecInferencer

    # model = TextRecInferencer(config_py, model_pth, device="cuda")

    # repo_files = [".gitattributes", "README.md", "config.json", "config.py", "dictionary.txt", "model.pth"]

    # config = "config.py"

    # dict_file = next((f for f in repo_files if any(f.endswith(ext) for ext in config)), None)

    # print(dict_file)

    # img = "/home/gabriel/Desktop/htrflow_core/data/demo_image.jpg"
    # image = cv2.imread(img)

    # results = model([image] * 100)
