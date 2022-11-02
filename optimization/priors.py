import math

import torch
import torch.nn as nn
from torchtyping import TensorType


class SpikeandSlab(nn.Module):
    def __init__(
        self,
        mixture: str,
        spike: float,
        slab: float,
        pi: float,
        n_objects: int,
        init_dim: int,
        device: torch.device,
    ) -> None:
        super(SpikeandSlab, self).__init__()
        self.mixture = mixture
        self.spike = spike
        self.slab = slab
        self.pi = pi
        self.n_objects = n_objects
        self.init_dim = init_dim
        self.device = device
        self.pdf = self.norm_pdf if self.mixture == "gaussian" else self.laplace_pdf
        self.initialize_priors_()

    @staticmethod
    def norm_pdf(
        X: TensorType["m", "d"], loc: TensorType["m", "d"], scale: TensorType["m", "d"]
    ) -> TensorType["m", "d"]:
        """Probability density function of a normal distribution."""
        return (
            torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2)))
            / scale
            * math.sqrt(2 * math.pi)
        )

    @staticmethod
    def laplace_pdf(
        X: TensorType["m", "d"], loc: TensorType["m", "d"], scale: TensorType["m", "d"]
    ) -> TensorType["m", "d"]:
        """Probability density function of a laplace distribution."""
        return torch.exp(-(X - loc).abs() / scale) / scale.mul(2.0)

    def forward(self, X: TensorType["m", "d"]) -> TensorType["m", "d"]:
        spike = self.pi * self.pdf(X, self.loc, self.scale_spike)
        slab = (1 - self.pi) * self.pdf(X, self.loc, self.scale_slab)
        return spike + slab

    def initialize_priors_(self) -> None:
        self.loc = torch.zeros(self.n_objects, self.init_dim).to(self.device)
        self.scale_spike = (
            torch.ones(self.n_objects, self.init_dim).mul(self.spike).to(self.device)
        )
        self.scale_slab = (
            torch.ones(self.n_objects, self.init_dim).mul(self.slab).to(self.device)
        )
