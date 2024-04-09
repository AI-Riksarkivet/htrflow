from typing import Optional

import torch


class PytorchDeviceMixin:
    @property
    def device(self) -> torch.device:
        return self.model.device

    def set_device(self, device: Optional[str] = None) -> torch.device:
        """Method to set the device for the model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def to(self, device: Optional[str]):
        """Move device to model.device"""
        self.model.to(device)

    def to_numpy(self, tensor):
        """
        Convert a PyTorch tensor to a NumPy array.
        Moves the tensor to CPU if it's on a GPU.
        """
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
