#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .trainer import Trainer
from typing import Any, Dict, Tuple

Array = Any
Tensor = Any
os.environ["PYTHONIOENCODING"] = "UTF-8"


class Sigma(nn.Module):
    def __init__(self, n_objects: int, init_dim: int, bias: bool = False):
        super(Sigma, self).__init__()
        self.logsigma = nn.Linear(n_objects, init_dim, bias=bias)

    def forward(self) -> Tensor:
        return self.logsigma.weight.T.exp()


class Mu(nn.Module):
    def __init__(self, n_objects: int, init_dim: int, bias: bool = False):
        super(Mu, self).__init__()
        self.mu = nn.Linear(n_objects, init_dim, bias=bias)
        # initialize means
        nn.init.kaiming_normal_(self.mu.weight, mode="fan_out", nonlinearity="relu")

    def forward(self) -> Tensor:
        return self.mu.weight.T


class VICE(Trainer):
    def __init__(
        self,
        n_train: int,
        n_objects: int,
        init_dim: int,
        optim: str,
        eta: str,
        batch_size: int,
        epochs: int,
        burnin: int,
        mc_samples: int,
        prior: str,
        spike: float,
        slab: float,
        pi: float,
        k: int,
        ws: int,
        steps: int,
        model_dir: str,
        results_dir: str,
        device: torch.device,
        verbose: bool = False,
        init_weights: bool = True,
        bias: bool = False,
    ):
        super().__init__(
            n_train=n_train,
            n_objects=n_objects,
            init_dim=init_dim,
            optim=optim,
            eta=eta,
            batch_size=batch_size,
            epochs=epochs,
            burnin=burnin,
            mc_samples=mc_samples,
            prior=prior,
            spike=spike,
            slab=slab,
            pi=pi,
            k=k,
            ws=ws,
            steps=steps,
            model_dir=model_dir,
            results_dir=results_dir,
            device=device,
            verbose=verbose,
        )
        self.mu = Mu(n_objects, init_dim, bias)
        self.sigma = Sigma(n_objects, init_dim, bias)

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def reparameterize(loc: Tensor, scale: Tensor) -> Tensor:
        """Apply reparameterization trick."""
        eps = scale.data.new(scale.size()).normal_()
        return eps.mul(scale).add(loc)

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu = self.mu()
        sigma = self.sigma()
        X = self.reparameterize(mu, sigma)
        z = F.relu(torch.mm(batch, X))
        return z, mu, sigma, X

    def _initialize_weights(self) -> None:
        # this is equivalent to 1 / std(mu)
        eps = -(self.mu.mu.weight.data.std().log() * -1.0).exp()
        nn.init.constant_(self.sigma.logsigma.weight, eps)

    @property
    def detached_params(self) -> Dict[str, Array]:
        """Detach params from computational graph."""
        loc = self.mu.mu.weight.data.T.detach()
        scale = self.sigma.logsigma.weight.data.T.exp().detach()
        params = dict(loc=loc.cpu().numpy(), scale=scale.cpu().numpy())
        return params
