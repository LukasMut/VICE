#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import unittest

import numpy as np
from main_tripletize import Tripletizer


test_dir = './test'
n_samples = int(1e+4)
M = 100
P = int(M/2)
k = 3
rnd_seed = 42

class TripletizeTestCase(unittest.TestCase):

    @staticmethod
    def generate_data(M, P) -> np.ndarray:
        X = np.random.randn(M, P)
        return X

    @staticmethod
    def save_data(X) -> np.ndarray:
        with open(os.path.join(test_dir, 'test_data.npy'), 'wb') as f:
            np.save(f, X)

    def test_triplet_generating(self):
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        X = self.generate_data(M, P)
        self.save_data(X)
        tripletizer = Tripletizer(
                                in_path=os.path.join(test_dir, 'test_data.npy'),
                                out_path=test_dir,
                                n_samples=n_samples,
                                rnd_seed=rnd_seed,
                                k=k
                                )
        triplets = tripletizer.sample_triplets()
        self.assertTrue(type(triplets) == np.ndarray)
        self.assertEqual(triplets.shape, (n_samples, k))
        tripletizer.save_triplets(triplets)

    def test_triplet_loading(self):
        train_triplets = np.load(os.path.join(test_dir, 'train_90.npy'))
        test_triplets = np.load(os.path.join(test_dir, 'test_10.npy'))
        self.assertTrue(type(train_triplets) == np.ndarray)
        self.assertTrue(type(test_triplets) == np.ndarray)
        self.assertEqual(train_triplets.shape[1], k)
        self.assertEqual(test_triplets.shape[1], k)
        shutil.rmtree(test_dir)




