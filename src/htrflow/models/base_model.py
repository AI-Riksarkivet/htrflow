from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseModel(ABC):
    """
    Model base class

    This is the abstract base class of HTRflow models. It handles batching
    of inputs, some shared initialization arguments and generic logging.

    Concrete model implementations bases this class and defines their
    prediction method in `_predict()`.
    """

    def __init__(self, device: str | None = None, allow_tf32: bool = True, allow_cudnn_benchmark: bool = False):
        """
        Arguments:
            device: Model device as a string, recognizable by torch. Defaults
                to `None`, which sets the device to `cuda` or `cpu` depending
                on availability.
            allow_tf32: Allow running matrix multiplications with TensorFloat-32.
                This speeds up inference at the expense of inference quality.
                On Ampere and newer CUDA devices, enabling TF32 can improve
                performance for matrix multiplications and convolutions.
                Read more here:
                https://huggingface.co/docs/diffusers/optimization/fp16#tensorfloat-32
            allow_cudnn_benchmark: When True, enables cuDNN benchmarking to
                select the fastest convolution algorithms for fixed input sizes,
                potentially increasing performance. Note that this may introduce
                nondeterminism. Defaults to False.
                Read more here:
                https://huggingface.co/docs/transformers/en/perf_train_gpu_one#tf32
        """
        self.metadata = {"model_class": self.__class__.__name__}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if torch.cuda.is_available():
            # Allow matrix multiplication with TensorFloat-32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = allow_cudnn_benchmark

    def predict(self, images: list[Image], **kwargs):
        """Perform inference on images

        Arguments:
            images: Input images
            **kwargs: Optional keyword arguments that are forwarded to
                the model specific prediction method `_predict(...)`.
        """
        return self._predict(images, **kwargs)

    @abstractmethod
    def _predict(self, images: list[Image], **kwargs):
        """Model specific prediction method"""

    def __call__(self, images: list[Image], **kwargs):
        """Alias for BaseModel.predict(...)"""
        return self.predict(images, **kwargs)
