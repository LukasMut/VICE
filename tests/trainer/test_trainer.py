#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import unittest
import utils

import numpy as np
import tests.helper as helper
import torch.nn as nn

from train.trainer import Trainer
from models.model import VICE, SPoSE

batch_size = 128
triplets = helper.create_triplets()
M = utils.get_nitems(triplets)

subsample = triplets[np.random.choice(triplets.shape[0], size=batch_size, replace=False)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainerTestCase(unittest.TestCase):


    def get_model(self):
        vice = VICE(prior=prior, in_size=M, out_size=int(M/2), init_weights=True)
        vice.to(device)
        return vice

    def test_properties(self):
        vice = self.get_model()
        triplets = helper.create_triplets()
        hypers = helper.get_hypers()
        trainer = Trainer(
                          model=vice,
                          task=hypers['task'],
                          N=hypers['N'],
                          n_items=hypers['M'],
                          latent_dim=hypers['P'],
                          optim=hypers['optim'],
                          eta=hypers['eta'],
                          batch_size=hypers['batch_size'],
                          epochs=hypers['epochs'],
                          mc_samples=hypers['mc_samples'],
                          prior=hypers['prior'],
                          spike=hypers['spike'],
                          slab=hypers['slab'],
                          pi=hypers['pi'],
                          steps=hypers['steps'],
                          device=device,
                          verbose=True,
        )
        self.assertEqual(trainer.model, vice)
        self.assertEqual(trainer.task, hypers['task'])
        self.assertEqual(trainer.N, hypers['N'])
        self.assertEqual(trainer.n_items, hypers['M'])
        self.assertEqual(trainer.latent_dim, hypers['P'])
        self.assertEqual(trainer.optim, hypers['optim'])
        self.assertEqual(trainer.eta, hypers['eta'])
        self.assertEqual(trainer.batch_size, hypers['batch_size'])
        self.assertEqual(trainer.epochs, hypers['epochs'])
        self.assertEqual(trainer.mc_samples, hypers['mc_samples'])
        self.assertEqual(trainer.prior, hypers['prior'])
        self.assertEqual(trainer.spike, hypers['spike'])
        self.assertEqual(trainer.slab, hypers['slab'])
        self.assertEqual(trainer.pi, hypers['pi'])
        self.assertEqual(trainer.steps, hypers['steps'])
        self.assertEqual(trainer.device, device)
        self.assertTrue(trainer.verbose)

        trainer.initialize_priors_()
        self.assertTrue(hasattr(trainer, 'loc'))
        self.assertTrue(hasattr(trainer, 'scale_spike'))
        self.assertTrue(hasattr(trainer, 'scale_slab'))
        
        self.assertEqual(trainer.loc, torch.zeros(hypers['M'], hypers['P'].to(device)))
        self.assertEqual(trainer.scale_spike, torch.zeros(hypers['M'], hypers['P']).mul(hypers['spike']).to(device))
        self.assertEqual(trainer.scale_slab, torch.zeros(hypers['M'], hypers['P']).mul(hypers['slab']).to(device))

        W_loc, W_scale = trainer.parameters()
        self.assertEqual(W_loc, F.relu(vice.encoder_mu.weight.data.T.cpu().numpy())
        self.assertEqual(W_loc, vice.encoder_mu.weight.data.T.exp()cpu().numpy())