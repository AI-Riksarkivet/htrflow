from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files

from htrflow_core.models.base_model import BaseModel


class UltralyticsBaseModel(BaseModel):
    SUPPORTED_MODEL_TYPES = (".pt", ".yaml")
    CONFIG_JSON = "config.json"
    META_MODEL_TYPE = "model"

    def _download_from_hub(
        self, hf_model_id: str, hf_token: Optional[str] = None, cache_dir: str | Path = "./.cache"
    ) -> Path:
        cache_dir = Path(cache_dir)
        repo_files = list_repo_files(repo_id=hf_model_id, repo_type=self.META_MODEL_TYPE, token=hf_token)

        if self.CONFIG_JSON in repo_files:
            _ = hf_hub_download(
                repo_id=hf_model_id,
                filename=self.CONFIG_JSON,
                repo_type=self.META_MODEL_TYPE,
                cache_dir=cache_dir,
                token=hf_token,
            )

        model_file = next((f for f in repo_files if any(f.endswith(ext) for ext in self.SUPPORTED_MODEL_TYPES)), None)

        if not model_file:
            raise ValueError(
                f"No model file of supported type: {self.SUPPORTED_MODEL_TYPES} found in repository {hf_model_id}."
            )

        return hf_hub_download(
            repo_id=hf_model_id,
            filename=model_file,
            repo_type=self.META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )
