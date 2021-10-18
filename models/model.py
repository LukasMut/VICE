#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.laplace import Laplace
from typing import Tuple


class VICE(nn.Module):

    def __init__(
        self,
        prior: str,
        in_size: int,
        out_size: int,
        init_weights: bool = True,
        bias: bool = False,
    ):
        super(VICE, self).__init__()
        self.prior = prior
        self.in_size = in_size
        self.out_size = out_size
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

    def forward(self, x: torch.Tensor,
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


class SPoSE(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        init_weights: bool = True,
    ):
        super(SPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def _initialize_weights(self) -> None:
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
