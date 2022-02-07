#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.trainer import Trainer
from typing import Dict, Tuple


os.environ['PYTHONIOENCODING'] = 'UTF-8'


class Sigma(nn.Module):

    def __init__(self, in_size: int, out_size: int, bias: bool):
        super(Sigma, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.logsigma = nn.Linear(
            self.in_size, self.out_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.logsigma(x)
        W_sigma = self.logsigma.weight.T.exp()
        return W_sigma


class Mu(nn.Module):

    def __init__(self, in_size: int, out_size: int, bias: bool):
        super(Mu, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.mu = nn.Linear(
            self.in_size, self.out_size, bias=bias)
        # initialize means
        nn.init.kaiming_normal_(
            self.mu.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.mu(x)
        W_mu = self.mu.weight.T
        return W_mu


class VICE(Trainer):

    def __init__(
        self,
        task: str,
        n_train: int,
        n_items: int,
        latent_dim: int,
        optim: str,
        eta: str,
        batch_size: int,
        epochs: int,
        mc_samples: int,
        prior: str,
        spike: float,
        slab: float,
        pi: float,
        steps: int,
        model_dir: str,
        results_dir: str,
        device: torch.device,
        temperature=None,
        verbose: bool = False,
        init_weights: bool = True,
        bias: bool = False,
    ):
        super().__init__(
            task,
            n_train,
            n_items,
            latent_dim,
            optim,
            eta,
            batch_size,
            epochs,
            mc_samples,
            prior,
            spike,
            slab,
            pi,
            steps,
            model_dir,
            results_dir,
            device,
            temperature,
            verbose,
        )
        self.in_size = n_items
        self.out_size = latent_dim
        self.mu = Mu(self.in_size, self.out_size, bias)
        self.sigma = Sigma(self.in_size, self.out_size, bias)

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def reparameterize(loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        eps = scale.data.new(scale.size()).normal_()
        W_sampled = eps.mul(scale).add(loc)
        return W_sampled

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        W_mu = self.mu(x)
        W_sigma = self.sigma(x)
        W_sampled = self.reparameterize(W_mu, W_sigma)
        z = torch.mm(x, W_sampled)
        z = F.relu(z)
        return z, W_mu, W_sigma, W_sampled

    def _initialize_weights(self) -> None:
        # this is equivalent to 1 / std(mu)
        eps = -(self.mu.mu.weight.data.std().log() * -1.0).exp()
        nn.init.constant_(self.sigma.logsigma.weight, eps)

    @property
    def detached_params(self) -> Dict[str, np.ndarray]:
        W_loc = self.mu.mu.weight.data.T.detach()
        W_scale = self.sigma.logsigma.weight.data.T.exp().detach()
        W_loc = F.relu(W_loc)
        params = {'loc': W_loc.cpu().numpy(), 'scale': W_scale.cpu().numpy()}
        return params
