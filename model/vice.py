#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .trainer import Trainer
from typing import (Any, Dict, Tuple)

Array = Any
Tensor = Any
os.environ["PYTHONIOENCODING"] = "UTF-8"


class Sigma(nn.Module):
    def __init__(self, n_objects: int, latent_dim: int, bias: bool = False):
        super(Sigma, self).__init__()
        self.logsigma = nn.Linear(n_objects, latent_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.logsigma(x)
        W_sigma = self.logsigma.weight.T.exp()
        return W_sigma


class Mu(nn.Module):
    def __init__(self, n_objects: int, latent_dim: int, bias: bool = False):
        super(Mu, self).__init__()
        self.mu = nn.Linear(n_objects, latent_dim, bias=bias)
        # initialize means
        nn.init.kaiming_normal_(self.mu.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        _ = self.mu(x)
        W_mu = self.mu.weight.T
        return W_mu


class VICE(Trainer):
    def __init__(
        self,
        task: str,
        n_train: int,
        n_objects: int,
        latent_dim: int,
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
            task=task,
            n_train=n_train,
            n_objects=n_objects,
            latent_dim=latent_dim,
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
        self.mu = Mu(n_objects, latent_dim, bias)
        self.sigma = Sigma(n_objects, latent_dim, bias)

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def reparameterize(loc: Tensor, scale: Tensor) -> Tensor:
        """Apply reparameterization trick."""
        eps = scale.data.new(scale.size()).normal_()
        W_sampled = eps.mul(scale).add(loc)
        return W_sampled

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        W_mu = self.mu(x)
        W_sigma = self.sigma(x)
        W_sampled = self.reparameterize(W_mu, W_sigma)
        z = F.relu(torch.mm(x, W_sampled))
        return z, W_mu, W_sigma, W_sampled

    def _initialize_weights(self) -> None:
        # this is equivalent to 1 / std(mu)
        eps = -(self.mu.mu.weight.data.std().log() * -1.0).exp()
        nn.init.constant_(self.sigma.logsigma.weight, eps)

    @property
    def detached_params(self) -> Dict[str, Array]:
        """Detach params from computational graph."""
        W_loc = self.mu.mu.weight.data.T.detach()
        W_scale = self.sigma.logsigma.weight.data.T.exp().detach()
        W_loc = F.relu(W_loc)
        params = dict(loc=W_loc.cpu().numpy(), scale=W_scale.cpu().numpy())
        return params
