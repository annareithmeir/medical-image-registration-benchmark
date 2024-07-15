import torch


def normalize_tensor_to_0_1(tensor: torch.tensor)-> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())