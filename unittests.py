#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import unittest
import utils
import torch

import numpy as np
import torch.nn as nn

from models.model import VSPoSE

NUM_SAMPLES = 1000
NUM_ITEMS = 20
BATCH_SIZE = 32
RND_SEED = 42

K = 3
P = 50


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = './test'

if not os.path.exists(TEST_DIR):
    print('\n...Creating test directory.\n')
    os.mkdir(TEST_DIR)


class ModelTestCase(unittest.TestCase):

    def test_model(self):
        model = VSPoSE(in_size=NUM_ITEMS, out_size=P, init_weights=True)
        model.to(DEVICE)

        self.assertTrue(issubclass(VSPoSE, nn.Module))
        self.assertTrue(
            hasattr(model, 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.assertTrue(hasattr(model, 'encoder_mu'))
        self.assertTrue(hasattr(model, 'encoder_logsigma'))

        self.assertEqual(
            len(torch.unique(model.encoder_logsigma.weight.data)), 1)
        self.assertTrue(model.encoder_logsigma.weight.data.sum() < 0.)
        self.assertTrue(model.encoder_logsigma.weight.data.exp().sum() > 0.)

        self.assertEqual(model.encoder_mu.in_features,
                         model.encoder_logsigma.in_features, NUM_ITEMS)
        self.assertEqual(model.encoder_mu.out_features,
                         model.encoder_logsigma.out_features, P)


class TripletLoadingTestCase(unittest.TestCase):

    def test_loading(self):
        """Test whether loading triplets from disk into memory works correctly."""
        train_triplets, test_triplets = utils.load_data(device=DEVICE, triplets_dir=TEST_DIR)
        self.assertEqual(train_triplets.shape[1], test_triplets.shape[1], K)
        self.assertTrue(isinstance(train_triplets, torch.Tensor))
        self.assertTrue(isinstance(test_triplets, torch.Tensor))

        # test whether number of items is correct
        n_items = utils.get_nitems(train_triplets)
        self.assertTrue(type(n_items) == int)
        self.assertEqual(n_items, NUM_ITEMS)

        # the following test depends on the variables above
        train_batches, test_batches = utils.load_batches(
                                                        train_triplets=train_triplets,
                                                        test_triplets=test_triplets,
                                                        n_items=n_items,
                                                        batch_size=BATCH_SIZE,
                                                        sampling_method='normal',
                                                        rnd_seed=RND_SEED,
        )

        self.assertEqual(len(train_batches), len(train_triplets) // BATCH_SIZE)
        self.assertEqual(len(test_batches), len(test_triplets) // BATCH_SIZE)


class MiniBatchingTestCase(unittest.TestCase):

    def test_batching(self):
        train_triplets, test_triplets = utils.load_data(device=DEVICE, triplets_dir=TEST_DIR)
        # test whether number of items is correct
        n_items = utils.get_nitems(train_triplets)
        # the following test depends on the variables above
        _, test_batches = utils.load_batches(
                                            train_triplets=train_triplets,
                                            test_triplets=test_triplets,
                                            n_items=n_items,
                                            batch_size=BATCH_SIZE,
                                            sampling_method='normal',
                                            rnd_seed=RND_SEED,
        )
        model = VSPoSE(in_size=NUM_ITEMS, out_size=P, init_weights=True)
        model.to(DEVICE)

        for batch in test_batches:

            self.assertTrue(isinstance(batch, torch.Tensor))
            self.assertEqual(batch.shape[0], BATCH_SIZE * K)
            self.assertEqual(batch.shape[1], NUM_ITEMS)

            batch = batch.to(DEVICE)
            self.assertEqual(batch.device, model.encoder_mu.weight.device,
                             model.encoder_logsigma.weight.device)

            out = model(batch)

            self.assertEqual(len(out), 4)

            for i, t in enumerate(out):
                self.assertTrue(isinstance(t, torch.Tensor))
                if i > 0:
                    self.assertEqual(
                        t.T.shape, model.encoder_mu.weight.shape, model.encoder_logsigma.weight.shape)

            self.assertTrue(torch.all(out[1] == model.encoder_mu.weight.T))
            self.assertTrue(
                torch.all(out[2] == model.encoder_logsigma.weight.exp().T))

            out = torch.unbind(torch.reshape(out[0], (-1, K, P)), dim=1)
            self.assertEqual(len(out), K)

            for t in out:
                self.assertEqual(t.shape, (BATCH_SIZE, P))


# def filter_triplets(triplets):
#     return np.array(list(filter(lambda x : len(set(x)) == len(x), triplets)))

def filter_triplets(triplets: np.ndarray) -> np.ndarray:
    def f(x: np.ndarray) -> int:
        return len(np.unique(x))
    return triplets[np.where(np.apply_along_axis(f, 1, triplets) == triplets.shape[1])[0]]


def save(split: np.ndarray, file_name: str) -> None:
    with open(os.path.join(TEST_DIR, file_name), 'wb') as f:
        np.save(f, split)


def create_triplets(train_frac: float = .9):
    a = np.random.randint(low=0, high=NUM_ITEMS, size=NUM_SAMPLES, dtype=int)
    b = a[np.random.permutation(len(a))]
    c = a[np.random.permutation(len(a))]
    triplets = np.c_[a, b, c]
    # discard triplets with duplicate items
    triplets = filter_triplets(triplets)
    rnd_perm = np.random.permutation(len(triplets))
    train_90 = triplets[rnd_perm[:int(len(rnd_perm) * train_frac)]]
    test_10 = triplets[rnd_perm[int(len(rnd_perm) * train_frac):]]
    # save triplet splits to disk
    save(train_90, 'train_90.npy')
    save(test_10, 'test_10.npy')


if __name__ == '__main__':
    create_triplets()
    unittest.main()
    shutil.rmtree(TEST_DIR)
