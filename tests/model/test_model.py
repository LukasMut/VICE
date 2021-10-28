#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import unittest
import utils

import numpy as np
import tests.helper as helper
import torch.nn as nn

from models.model import VICE, SPoSE

batch_size = 128
triplets = helper.create_triplets()
subsample = triplets[np.random.choice(triplets.shape[0], size=batch_size, replace=False)]
M = utils.get_nitems(triplets)
prior = 'gaussian'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VICETestCase(unittest.TestCase):

    def get_model(self):
        vice = VICE(prior=prior, in_size=M, out_size=int(M/2), init_weights=True)
        vice.to(device)
        return vice

    def test_attributes(self):
        vice = self.get_model()            
        self.assertTrue(issubclass(VICE, nn.Module))
        self.assertTrue(
            hasattr(vice, 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.assertTrue(hasattr(vice, 'encoder_mu'))
        self.assertTrue(hasattr(vice, 'encoder_logsigma'))
        self.assertTrue(vice.prior, 'gaussian')

        self.assertEqual(
            len(torch.unique(vice.encoder_logsigma.weight.data)), 1)
        
        self.assertTrue(vice.encoder_logsigma.weight.data.sum() < 0.)
        self.assertTrue(vice.encoder_logsigma.weight.data.exp().min() >= 0.)

        self.assertEqual(vice.encoder_mu.in_features,
                         vice.encoder_logsigma.in_features, M)
        
        self.assertEqual(vice.encoder_mu.out_features,
                         vice.encoder_logsigma.out_features, int(M/2))
    
    def test_output(self):
        vice = self.get_model()
        batches = utils.load_batches(train_triplets=None, test_triplets=subsample, n_items=M, batch_size=batch_size, inference=True)
        for batch in batches:
            out = vice(batch)
            self.assertEqual(len(out), 4)
            z, W_mu, W_sigma, W_sample = out
            self.assertTrue(z.min() >= 0.)
            self.assertTrue(W_sigma >= 0.)
            self.assertEqual(W_mu.shape, W_sigma.shape, W_sample.shape)
            self.assertEqual(z.shape, (int(batch_size * 3), M))


class SPoSETestCase(unittest.TestCase):

    def get_model(self):
        spose = SPoSE(in_size=M, out_size=int(M/2), init_weights=True)
        spose.to(device)
        return spose

    def test_attributes(self):
        spose = self.get_model()            
        self.assertTrue(issubclass(SPoSE, nn.Module))
        self.assertTrue(
            hasattr(spose, 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.assertTrue(hasattr(spose, 'fc'))

        self.assertEqual(spose.fc.in_features, M)
        self.assertEqual(spose.fc.out_features, int(M/2))

    def test_output(self):
        spose = self.get_model()
        batches = utils.load_batches(train_triplets=None, test_triplets=subsample, n_items=M, batch_size=batch_size, inference=True)
        for batch in batches:
            out = spose(batch)
            self.assertEqual(out.shape, (int(batch_size * 3), M))


