#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.trainer import Trainer
from torch.distributions.laplace import Laplace
from typing import Dict, Tuple


os.environ['PYTHONIOENCODING'] = 'UTF-8'


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
        self.prior = prior
        self.in_size = n_items
        self.out_size = latent_dim
        self.encoder_mu = nn.Linear(self.in_size, self.out_size, bias=bias)
        self.encoder_logsigma = nn.Linear(
            self.in_size, self.out_size, bias=bias)

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def reparameterize(prior: str, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if prior == 'gaussian':
            eps = scale.data.new(scale.size()).normal_()
            W_sampled = eps.mul(scale).add(loc)
        else:
            laplace = Laplace(loc=torch.zeros(scale.size()),
                              scale=torch.ones(scale.size()))
            U = laplace.sample().to(scale.device)
            W_sampled = U.mul(scale).add(loc)
        return W_sampled

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = self.encoder_mu(x)
        _ = self.encoder_logsigma(x)
        W_mu = self.encoder_mu.weight.T
        W_sigma = self.encoder_logsigma.weight.T.exp()
        W_sampled = self.reparameterize(self.prior, W_mu, W_sigma)
        z = torch.mm(x, W_sampled)
        z = F.relu(z)
        return z, W_mu, W_sigma, W_sampled

    def _initialize_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if re.search(r'mu', n):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # this is equivalent to but numerically more stable than mu_std.pow(-1)
                    eps = (self.encoder_mu.weight.data.std().log() * -1.0).exp()
                    nn.init.constant_(m.weight, -eps)

    @property
    def detached_params(self) -> Dict[str, np.ndarray]:
        W_loc = self.encoder_mu.weight.data.T.detach()
        W_scale = self.encoder_logsigma.weight.data.T.exp().detach()
        W_loc = F.relu(W_loc)
        params = {'loc': W_loc.cpu().numpy(), 'scale': W_scale.cpu().numpy()}
        return params
