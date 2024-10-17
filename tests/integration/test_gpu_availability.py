import pytest
import torch


@pytest.mark.gpu
def test_gpu_availability():
    assert torch.cuda.is_available(), "CUDA GPU is not available"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
