#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import torch
import unittest

import numpy as np
import tests.helper as helper
import torch.nn.functional as F

from train.trainer import Trainer
from models.model import VICE

test_dir = './test'
model_dir = os.path.join(test_dir, 'model')
hypers = helper.get_hypers()
triplets = helper.create_triplets()
subsample = triplets[np.random.choice(triplets.shape[0], size=hypers['batch_size'], replace=False)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainerTestCase(unittest.TestCase):


    def get_model(self, hypers):
        vice = VICE(prior=hypers['prior'], in_size=hypers['M'], out_size=hypers['P'], init_weights=True)
        vice.to(device)
        return vice

    def test_properties(self):
        triplets = helper.create_triplets()        
        vice = self.get_model(hypers)
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
                          model_dir=model_dir,
                          results_dir=test_dir,
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
        
        np.testing.assert_allclose(trainer.loc, torch.zeros(hypers['M'], hypers['P']).to(device))
        np.testing.assert_allclose(trainer.scale_spike, torch.ones(hypers['M'], hypers['P']).mul(hypers['spike']).to(device))
        np.testing.assert_allclose(trainer.scale_slab, torch.ones(hypers['M'], hypers['P']).mul(hypers['slab']).to(device))

        W_loc, W_scale = trainer.parameters
        np.testing.assert_allclose(W_loc, F.relu(vice.encoder_mu.weight.data.T.cpu()).numpy())
        np.testing.assert_allclose(W_scale, vice.encoder_logsigma.weight.data.T.exp().cpu().numpy())