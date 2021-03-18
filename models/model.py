#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['VSPoSE']

import os
import re
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.laplace import Laplace
from typing import Tuple

class VSPoSE(nn.Module):

    def __init__(self, in_size:int, out_size:int, init_weights:bool=True, bias:bool=True):
        super(VSPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.decode = decode
        self.encoder_mu = nn.Sequential(
                                        nn.Linear(self.in_size, self.out_size, bias=bias),
                                        nn.Identity(),
                                        )
        #b is constrained to be in the positive real number space R+ (subject to b > 0)
        self.encoder_b = nn.Sequential(
                                        nn.Linear(self.in_size, self.out_size, bias=bias),
                                        nn.Softplus(),
                                        )
        if init_weights:
            self._initialize_weights()

    def reparameterize(self, mu:torch.Tensor, b:torch.Tensor, device:torch.device) -> torch.Tensor:
        laplace = Laplace(loc=torch.zeros(b.size()), scale=torch.ones(b.size()))
        U = laplace.sample().to(device) #draw random sample from a standard Laplace distribution with mu = 0 and lam = 1
        z = U.mul(b).add(mu) #perform reparameterization trick
        return z

    def forward(self, x:torch.Tensor, device:torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.encoder_mu(x) #mu = mode = loc param
        b = self.encoder_b(x) #b = scale param
        z = self.reparameterize(mu, b, device)
        z = F.relu(z) #employ rectifier to impose non-negativity constraint on z (move all negative numbers into null space)
        l = (b.log()*-1.0).exp() #this is equivalent to but numerically more stable than b.pow(-1)
        return z, mu, l

    def _initialize_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if re.search(r'mu', n):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.constant_(m.weight, float(self.encoder_mu[0].weight.data.std()/1e+3))
                    if hasattr(m.bias, 'data'):
                        nn.init.constant_(m.bias, 0)
