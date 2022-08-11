from typing import Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


class TripletLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super(TripletLoss, self).__init__()
        self.temperature = temperature

    def logsumexp(self, dots: Tuple[Tensor]) -> Tensor:
        return torch.log(
            torch.sum(torch.exp(torch.stack(dots) / self.temperature), dim=0)
        )

    def log_softmax(self, dots: Tuple[Tensor]) -> Tensor:
        return dots[0] / self.temperature - self.logsumexp(dots)

    def cross_entropy_loss(self, dots: Tuple[Tensor]) -> Tensor:
        return torch.mean(-self.log_softmax(dots))

    def forward(self, dots: Tuple[Tensor]):
        return self.cross_entropy_loss(dots)
