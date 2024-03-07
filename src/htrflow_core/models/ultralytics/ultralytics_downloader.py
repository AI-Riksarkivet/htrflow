from huggingface_hub import hf_hub_download, list_repo_files

from htrflow_core.models.hf_downloader import HFDownloader


class UltralyticsDownloader(HFDownloader):
    SUPPORTED_MODEL_TYPES = (".pt", ".yaml")

    @classmethod
    def _download_from_hub(cls, hf_model_id, hf_token, cache_dir):
        repo_files = list_repo_files(repo_id=hf_model_id, repo_type=cls.META_MODEL_TYPE, token=hf_token)

        if cls.CONFIG_JSON in repo_files:
            _ = hf_hub_download(
                repo_id=hf_model_id,
                filename=cls.CONFIG_JSON,
                repo_type=cls.META_MODEL_TYPE,
                cache_dir=cache_dir,
                token=hf_token,
            )

        model_file = next((f for f in repo_files if any(f.endswith(ext) for ext in cls.SUPPORTED_MODEL_TYPES)), None)

        if not model_file:
            raise ValueError(
                f"No model file of supported type: {cls.SUPPORTED_MODEL_TYPES} found in repository {hf_model_id}."
            )

        return hf_hub_download(
            repo_id=hf_model_id,
            filename=model_file,
            repo_type=cls.META_MODEL_TYPE,
            cache_dir=cache_dir,
            token=hf_token,
        )


if __name__ == "__main__":
    model = UltralyticsDownloader.from_pretrained(model="ultralyticsplus/yolov8s")
