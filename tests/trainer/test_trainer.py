#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import json
import shutil
import torch
import unittest
import utils

import numpy as np
import tests.helper as helper
import torch.nn as nn
import torch.nn.functional as F

from train.trainer import Trainer
from models.model import VICE
from typing import Any

test_dir = './test'
model_dir = os.path.join(test_dir, 'model')
hypers = helper.get_hypers()
triplets = helper.create_triplets()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainerTestCase(unittest.TestCase):

    def get_model(self, hypers: dict) -> Any:
        vice = VICE(prior=hypers['prior'], in_size=hypers['M'], out_size=hypers['P'], init_weights=True)
        vice.to(device)
        return vice
    
    def get_trainer(self, model: nn.Module, hypers: dict) -> object:
        trainer = Trainer(
                          model=model,
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
        return trainer
        
    def test_properties(self) -> None:     
        vice = self.get_model(hypers)
        trainer = self.get_trainer(vice, hypers)
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

    def test_optimization(self) -> None:
        vice = self.get_model(hypers)
        trainer = self.get_trainer(vice, hypers)
        # get model parameters at initilization
        W_loc_init, W_scale_init = trainer.parameters
        
        train_triplets, test_triplets = helper.create_train_test_split(triplets)
        train_batches, val_batches = utils.load_batches(train_triplets=train_triplets, test_triplets=test_triplets, 
                    n_items=hypers['M'], batch_size=hypers['batch_size'], inference=False)
        
        trainer.train(train_batches=train_batches, val_batches=val_batches)
        # get model paramters after optimization
        W_loc_opt, W_scale_opt = trainer.parameters

        # check whether model parameters have changed during optimization
        self.assertTrue(self.assert_difference(W_loc_init, W_loc_opt))
        self.assertTrue(self.assert_difference(W_scale_init, W_scale_opt))


    @staticmethod
    def assert_difference(A: np.ndarray, B: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(A, B)
            return False
        except:
            return True


    def test_results(self) -> None:
        results = []
        regex = r'(?=.*\d)(?=.*json$)'
        for root, _, files in os.walk(test_dir):
            for f in files:
                if re.compile(regex).search(f):
                    with open(os.path.join(root, f), 'r') as rf:
                        r = json.load(rf)
                    self.assertTrue(isinstance(r, dict))
                    results.append(r)
        self.assertEqual(len(results), int(hypers['epochs'] / hypers['steps']))
        shutil.rmtree(test_dir)
        