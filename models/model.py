#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from torch.distributions.laplace import Laplace
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
        #NOTE: b is constrained to be in the positive real number space R+ which is why we code for log(b) (recall that exp(log(b)) is subject to R+)
        self.encoder_logb = nn.Linear(self.in_size, self.out_size, bias=bias)

        if init_weights:
            self._initialize_weights()

    def reparameterize(self, mu:torch.Tensor, b:torch.Tensor, device:torch.device) -> torch.Tensor:
        laplace = Laplace(loc=torch.zeros(b.size()), scale=torch.ones(b.size()))
        U = laplace.sample().to(device) #draw random sample from a standard Laplace distribution with mu = 0 and lam = 1
        z = U.mul(b).add(mu) #perform reparameterization trick
        return z

    def forward(self, x:torch.Tensor, device:torch.device) -> Tuple[torch.Tensor]:
        mu = self.encoder_mu(x)
        log_b = self.encoder_logb(x)
        b = log_b.exp()
        z = self.reparameterize(mu, b, device)
        z = F.relu(z) #employ rectifier to impose non-negativity constraint on z (move all negative numbers into null space)
        l = (log_b*-1.0).exp() #this is equivalent to but numerically more stable than b.pow(-1)
        return z, mu, l

    def _initialize_weights(self) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if re.search(r'mu', n):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    eps = (self.encoder_mu.weight.data.std().log()*-1.0).exp() #this is equivalent to but numerically more stable than mu_std.pow(-1)
                    nn.init.constant_(m.weight, -eps)
