#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class VSPoSE(nn.Module):

    def __init__(
                self,
                in_size:int,
                out_size:int,
                init_weights:bool=True,
                bias:bool=False,
                ):
        super(VSPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.encoder_mu = nn.Linear(self.in_size, self.out_size, bias=bias)
        self.encoder_logsigma = nn.Linear(self.in_size, self.out_size, bias=bias)

        if init_weights:
            self._initialize_weights()

    def reparameterize(self, mu:torch.Tensor, sigma:torch.Tensor) -> torch.Tensor:
        eps = sigma.data.new(sigma.size()).normal_()
        return eps.mul(sigma).add_(mu)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.encoder_mu(x)
        log_sigma = self.encoder_logsigma(x)
        W_mu = self.encoder_mu.weight.T
        W_sigma = self.encoder_logsigma.weight.T.exp()
        W_sampled = self.reparameterize(W_mu, W_sigma)
        z = torch.mm(x, W_sampled)
        z = F.relu(z) #employ rectifier to impose non-negativity constraint on z (move all negative numbers into null space)
        return z, W_mu, W_sigma, W_sampled

    def _initialize_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if re.search(r'mu', n):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    eps = (self.encoder_mu.weight.data.std().log()*-1.0).exp() #this is equivalent to but numerically more stable than mu_std.pow(-1)
                    nn.init.constant_(m.weight, -eps)
