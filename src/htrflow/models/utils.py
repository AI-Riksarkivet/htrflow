import torch


def check_device_to_use(device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    return device
