import torch
import torch.nn as nn
from torch import Tensor


def freeze_params(module: nn.Module) -> None:
    for _, p in module.named_parameters():
        p.requires_grad = False


def subsequent_mask(size: int) -> Tensor:
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)


class ConfigurationError(Exception):
    """Custom exception for misspecifications of configuration."""
