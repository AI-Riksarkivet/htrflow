import logging
import os

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from mmdet.apis import DetInferencer
from mmengine.config import Config
from mmocr.apis import TextRecInferencer

from htrflow.models.base_model import BaseModel
from htrflow.models.framework_enums import ModelFrameworks


class OpenmmlabModelLoader:
    REPO_TYPE = "model"
    MODEL_FILE = "model.pth"
    CONFIG_FILE = "config.py"
    DICT_FILE = "dictionary.txt"

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: str = None, device: str = None):
        device = cls.check_device_to_use(device)

        model_file, config_file = cls._download_config_and_model_file(model_id, cache_dir)

        if model_file and config_file:
            model_scope, config_file = cls._checking_model_scope(model_id, cache_dir, config_file)

            model = OpenModelFactory.create_openmmlab_model(model_scope, config_file, model_file, device)
            return model
        return None

    @classmethod
    def check_device_to_use(cls,device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device if torch.cuda.is_available() else "cpu")
        return device


    @classmethod
    def _checking_model_scope(cls, model_id, cache_dir, config_file):

        cfg = Config.fromfile(config_file)

        model_scope = cfg.default_scope

        if model_scope == OpenmmlabsFramework.MMOCR.value:
            download_dict_file = cls._download_dictonary_file(model_id, cache_dir)

            if os.path.exists(config_file):
                os.remove(config_file)

            cfg.dictionary["dict_file"] = download_dict_file
            cfg.model["decoder"]["dictionary"]["dict_file"] = download_dict_file
            cfg.dump(config_file)

        return model_scope, config_file


    @classmethod
    def _download_config_and_model_file(cls, repo_id, cache_dir):
        try:
            model_file = hf_hub_download(
                repo_id=repo_id,
                repo_type=cls.REPO_TYPE,
                filename=cls.MODEL_FILE,
                library_name=__package__,
                cache_dir=cache_dir,
            )
            config_file = hf_hub_download(
                repo_id=repo_id,
                repo_type=cls.REPO_TYPE,
                filename=cls.CONFIG_FILE,
                library_name=__package__,
                cache_dir=cache_dir,
            )
            return model_file, config_file

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


class OpenModelFactory:
    @staticmethod
    def create_openmmlab_model(model_scope, config_file, model_file, device):
        model_creators = {
            ModelFrameworks.MMDET.value: DetInferencer,
            ModelFrameworks.MMOCR.value: TextRecInferencer,
        }

        if model_scope in model_creators:
            return OpenmmlabModel( model_creators[model_scope] , model_scope,config_file, model_file, device)

        logging.error(f"Unknown model scope: {model_scope}")
        return None



class OpenmmlabModel(BaseModel):
    def __init__(self, inferencer ,framework ,config_file, model_file, device):
        self.model = inferencer(config_file, model_file, device)
        self.framework = framework
        self.device = device

    def __str__(self) -> str:
        return f"{self.model.__str__()}"


    # TODO this should work on this here
    # TODO Look at docuemntation for different init and args on call:
    # TODO Update unit test --> However use unittest.mocks

    #     @classmethod
    # def from_local(cls, config_file: str, model_files, device: str = None):
    #     cfg = Config.fromfile(config_file)
    #     model = OpenModelFactory.create_openmmlab_model(cfg, config_file, model_files, device)
    #     return model



if __name__ == "__main__":
    region_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="/home/gabriel/Desktop/htrflow_core/models")
    text_model = OpenmmlabModelLoader.from_pretrained("Riksarkivet/satrn_htr", cache_dir="/home/gabriel/Desktop/htrflow_core/models")

    print(region_model)
    print(region_model.framework)
    print(text_model.device)


    print(text_model)
    print(text_model.framework)
    print(text_model.device)

