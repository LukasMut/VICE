#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
            'SPoSE',
            'VSPoSE',
            'l1_regularization',
            ]

import os
import re
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as pjoin
from torch.distributions.laplace import Laplace
from typing import Tuple
from utils import load_model

class VSPoSE(nn.Module):

    def __init__(
                self,
                in_size:int,
                out_size:int,
                init_weights:bool=True,
                init_method:str='normal',
                bias:bool=True,
                decode:bool=False,
                device=None,
                rnd_seed=None,
                dir=None,
                ):
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
        if self.decode:
            self.decoder = nn.Linear(self.out_size, self.out_size, bias=bias)

        if init_weights:
            #initialise weights of model with converged deterministic SPoSE model or very small values following a normal distribution
            init_methods = ['dSPoSE', 'nmf', 'normal']
            assert init_method in init_methods, f'initialization method must be one of {init_methods}'
            self.init_method = init_method
            self._initialize_weights(device=device, rnd_seed=rnd_seed, dir=dir)

    def reparameterize(self, mu:torch.Tensor, b:torch.Tensor, device:torch.device) -> torch.Tensor:
        laplace = Laplace(loc=torch.zeros(b.size()), scale=torch.ones(b.size()))
        U = laplace.sample().to(device) #draw random sample from a standard Laplace distribution with mu = 0 and lam = 1
        z = U.mul(b).add(mu) #perform reparameterization trick
        return z

    def forward(self, x:torch.Tensor, device:torch.device) -> Tuple[torch.Tensor]:
        mu = self.encoder_mu(x) #mu = mode = loc param
        b = self.encoder_b(x) #b = scale param
        #b = log_b.exp()
        z = self.reparameterize(mu, b, device)
        z = F.relu(z) #employ rectifier to impose non-negativity constraint on z (move all negative numbers into null space)
        if self.decode:
            z = self.decoder(z)
        l = (b.log()*-1.0).exp() #this is equivalent to but numerically more stable than b.pow(-1)
        return z, mu, l

    def _initialize_weights(self, device=None, rnd_seed=None, dir=None) -> None:
        if self.init_method == 'dSPoSE':
            lmbda = 0.008
            dSPoSE = SPoSE(in_size=self.in_size, out_size=self.out_size, init_weights=True)
            dSPoSE = load_model(
                                model=dSPoSE,
                                results_dir='./results',
                                modality='behavioral',
                                version='deterministic',
                                data='human',
                                dim=self.out_size,
                                lmbda=lmbda,
                                rnd_seed=rnd_seed,
                                device=device,
                                )
            W_dspose = dSPoSE.fc.weight.data
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if re.search(r'mu', n):
                        m.weight = W_dspose
                    else:
                        nn.init.constant_(m.weight, 1e-5)
                    if hasattr(m.bias, 'data'):
                        nn.init.constant_(m.bias, 0)
        elif self.init_method == 'nmf':
            assert type(dir) == str, '\nTo initialize V-SPoSE with NMF components, directory where to load W from must be provided.\n'
            with open(pjoin(dir, 'nmf_components.npy'), 'rb') as f:
                W_nmf = np.load(f)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if re.search(r'mu', n):
                        assert m.weight.data.shape == W_nmf.shape, '\nW_mu and W_nmf must be of equal shape.\n'
                        m.weight.data = torch.from_numpy(W_nmf)
                    else:
                        nn.init.constant_(m.weight, 1e-5)
                    if hasattr(m.bias, 'data'):
                        nn.init.constant_(m.bias, 0)
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if re.search(r'mu', n):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.constant_(m.weight, float(self.encoder_mu[0].weight.data.std()/1e+3))
                        if hasattr(m.bias, 'data'):
                            nn.init.constant_(m.bias, 0)
class SPoSE(nn.Module):

    def __init__(
                self,
                in_size:int,
                out_size:int,
                init_weights:bool=True,
                ):
        super(SPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def _initialize_weights(self) -> None:
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

def l1_regularization(model) -> torch.Tensor:
    l1_reg = torch.tensor(0., requires_grad=True)
    for n, p in model.named_parameters():
        if re.search(r'weight', n):
            l1_reg = l1_reg + torch.norm(p, 1)
    return l1_reg
