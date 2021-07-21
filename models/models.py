from typing import List, Tuple
import torch
from torch import nn
from torch.nn.modules.linear import Linear

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import optimizer


class LinearModel(pl.LightningModule):
    """
    The model parameters are come from ->
    https://arxiv.org/pdf/1711.07128v3.pdf
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

    def training_step(self, batch, batch_idx) -> None:
        # this is only for training
        x, y = batch
        x = self.flatten(x)
        y_hat = self.linear(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.flatten(x)
        y_hat = self.linear(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e3)
        return optimizer