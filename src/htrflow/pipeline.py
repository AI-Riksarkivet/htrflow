from collections import OrderedDict

import torch
import yaml

from htrflow.models.openmmlab_models import OpenmmlabModel


class MultiModelManager:
    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path
        self.models = OrderedDict()  # Holds the loaded models

    def load_models(self):
        with open(self.yaml_file_path, "r") as file:
            model_definitions = yaml.safe_load(file)
            for model_def in model_definitions:
                model_id = model_def.get("model_id")
                cache_dir = model_def.get("cache_dir")
                device = model_def.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                model_manager = OpenmmlabModel.from_pretrained(model_id, cache_dir, device)
                self.models[model_id] = model_manager
