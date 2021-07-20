from typing import List, Tuple
import torch
from torch import nn
from torch.nn.modules.linear import Linear

"""
    The model parameters are come from ->
    https://arxiv.org/pdf/1711.07128v3.pdf

"""


class LinearModel(nn.Module):
    def __init__(self, input_size: Tuple[int], len_classes: int = None) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear((input_size[0] * input_size[1]), 144),
            nn.ReLU(),
            nn.Linear(144, 144),
            nn.ReLU(),
            nn.Linear(144, 144),
            nn.ReLU(),
            nn.Linear(144, len_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits
