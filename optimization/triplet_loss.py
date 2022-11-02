from typing import List

import torch
import torch.nn as nn
from torchtyping import TensorType


class TripletLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super(TripletLoss, self).__init__()
        self.temperature = temperature

    def logsumexp(self, dots: List[TensorType["b"]]) -> TensorType["b"]:
        return torch.log(
            torch.sum(torch.exp(torch.stack(dots) / self.temperature), dim=0)
        )

    def log_softmax(self, dots: List[TensorType["b"]]) -> TensorType["b"]:
        return dots[0] / self.temperature - self.logsumexp(dots)

    def cross_entropy_loss(self, dots: List[TensorType["b"]]) -> TensorType["1"]:
        return torch.mean(-self.log_softmax(dots))

    def forward(self, dots: List[TensorType["b"]]) -> TensorType["1"]:
        return self.cross_entropy_loss(dots)
