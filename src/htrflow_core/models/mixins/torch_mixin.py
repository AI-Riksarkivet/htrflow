from typing import Optional

import numpy as np
import torch

from htrflow_core.models.enums import Framework


class PytorchMixin:
    def set_device(self, device: Optional[str] = None) -> torch.device:
        """Method to set the device for the model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            device_id = torch.cuda.current_device()
        else:
            device_id = "cpu"

        self.device = device
        self.device_id = device_id
        return torch.device(device)

    def to(self, device: Optional[str]) -> None:
        """Move device to model.device"""
        if self.framework == Framework.Openmmlab.value:
            self.move_openmmlab_device()

        self.metadata.update({"device": device})
        self.device = device
        self.model.to(device)

    def move_openmmlab_device(self) -> None:
        raise NotImplementedError

    def to_numpy(self, tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy array.
        Moves the tensor to CPU if it's on a GPU.
        """
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
