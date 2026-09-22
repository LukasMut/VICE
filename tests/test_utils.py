#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import shutil
import torch
import unittest
import utils
import warnings

import numpy as np
import helper
import torch.nn as nn


test_dir = './test'
train_file = 'train_90.npy'
test_file = 'test_10.npy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TripletLoadingTestCase(unittest.TestCase):

    @staticmethod
    def save_triplets(train_triplets, test_triplets) -> None:
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        with open(os.path.join(test_dir, train_file), 'wb') as f:
            np.save(f, train_triplets)
        with open(os.path.join(test_dir, test_file), 'wb') as f:
            np.save(f, test_triplets)

    def test_loading(self) -> None:
        triplets = helper.create_triplets()
        train_triplets_before, test_triplets_before = helper.create_train_test_split(triplets)
        self.save_triplets(train_triplets_before, test_triplets_before)
        train_triplets_after, test_triplets_after = utils.load_data(device=device, triplets_dir=test_dir)
        shutil.rmtree(test_dir)
        
        np.testing.assert_allclose(train_triplets_before, train_triplets_after)
        np.testing.assert_allclose(test_triplets_before, test_triplets_after)

        train_triplets = train_triplets_after
        test_triplets = test_triplets_after

        self.assertEqual(train_triplets.shape[1], test_triplets.shape[1])
        self.assertEqual(type(train_triplets), type(test_triplets))
        self.assertTrue(isinstance(train_triplets, torch.Tensor))
        self.assertTrue(isinstance(test_triplets, torch.Tensor))

        M = utils.get_nobjects(train_triplets)
        self.assertTrue(type(M) == int)

        hypers = helper.get_hypers()
        self.assertEqual(M, hypers['M'])


class DuplicateTripletsTestCase(unittest.TestCase):

    @staticmethod
    def save_triplets(train_triplets, test_triplets) -> None:
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        with open(os.path.join(test_dir, train_file), 'wb') as f:
            np.save(f, train_triplets)
        with open(os.path.join(test_dir, test_file), 'wb') as f:
            np.save(f, test_triplets)

    @staticmethod
    def duplicate_rows(triplets, n_repeats: int):
        """Append the first <n_repeats> rows a second time."""
        return np.vstack((triplets, triplets[:n_repeats]))

    def test_counting(self) -> None:
        triplets = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2], [0, 1, 2], [6, 7, 8]])
        n_duplicated_rows, n_duplicated_triplets = utils.count_duplicates(triplets)
        self.assertEqual(n_duplicated_rows, 3)
        self.assertEqual(n_duplicated_triplets, 1)

    def test_counting_without_duplicates(self) -> None:
        triplets = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(utils.count_duplicates(triplets), (0, 0))

    def test_permuted_rows_are_not_duplicates(self) -> None:
        # the column order encodes the choice, hence a permuted row is a different trial
        triplets = torch.tensor([[0, 1, 2], [2, 1, 0]])
        self.assertEqual(utils.count_duplicates(triplets), (0, 0))

    @staticmethod
    def load_and_catch_warnings(n_objects):
        """Load from test_dir, returning every duplicate warning it raised."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            loaded = utils.load_data(
                device=device, triplets_dir=test_dir, n_objects=n_objects)
        return loaded, [
            w for w in caught if issubclass(w.category, utils.DuplicateTripletsWarning)
        ]

    # helper.create_triplets is unseeded, so n_objects is passed explicitly
    # everywhere below: inferring it would make C(M,3), and hence every
    # p-value in these tests, a random variable across runs.

    def test_warning(self) -> None:
        triplets = helper.create_triplets(N=100, M=20, P=10)
        train_triplets, test_triplets = helper.create_train_test_split(triplets)
        n_repeats = 20
        train_triplets = self.duplicate_rows(train_triplets.numpy(), n_repeats)
        self.save_triplets(train_triplets, test_triplets)

        with self.assertWarns(utils.DuplicateTripletsWarning) as cm:
            loaded_train, _ = utils.load_data(
                device=device, triplets_dir=test_dir, n_objects=20)
        shutil.rmtree(test_dir)

        self.assertIn(f'Found {n_repeats * 2} of', str(cm.warning))
        self.assertIn(train_file.split('.')[0], str(cm.warning))
        self.assertIn('expected', str(cm.warning))
        # the warning must not drop or alter any of the loaded triplets
        np.testing.assert_allclose(train_triplets, loaded_train)

    def test_no_warning(self) -> None:
        triplets = helper.create_triplets(N=100, M=20, P=10)
        train_triplets, test_triplets = helper.create_train_test_split(triplets)
        self.save_triplets(train_triplets, test_triplets)

        _, duplicate_warnings = self.load_and_catch_warnings(n_objects=20)
        shutil.rmtree(test_dir)

        self.assertEqual(len(duplicate_warnings), 0)

    def test_repeats_at_the_chance_rate_are_silent(self) -> None:
        """The point of the feature: 200 draws from 20 possible triplets are
        almost all repeats, and every one of them is expected."""
        rnd = np.random.RandomState(0)
        cells = np.array(list(itertools.combinations(range(6), 3)))
        train_triplets = cells[rnd.randint(0, len(cells), size=200)]
        self.save_triplets(train_triplets, cells[:5])

        loaded, duplicate_warnings = self.load_and_catch_warnings(n_objects=6)
        shutil.rmtree(test_dir)

        # the data really is almost entirely duplicates ...
        n_duplicated_rows, _ = utils.count_duplicates(loaded[0])
        self.assertGreater(n_duplicated_rows, 150)
        # ... and none of it is surprising
        self.assertEqual(len(duplicate_warnings), 0)

    def test_self_concatenated_file_warns_with_a_usable_p_value(self) -> None:
        # 240 distinct triplets doubled puts the tail past 1e-308, which is
        # exactly the case the log10 fallback exists for
        triplets = helper.create_triplets(N=300, M=60, P=20)
        train_triplets, test_triplets = helper.create_train_test_split(triplets)
        train_triplets = train_triplets.numpy()
        self.save_triplets(np.vstack((train_triplets, train_triplets)), test_triplets)

        _, duplicate_warnings = self.load_and_catch_warnings(n_objects=60)
        shutil.rmtree(test_dir)

        self.assertEqual(len(duplicate_warnings), 1)
        message = str(duplicate_warnings[0].message)
        # p underflows float64 this far into the tail, so the message must
        # carry log10 p rather than a useless 'p = 0'
        self.assertIn('log10 p', message)
        self.assertNotIn('p = 0.0', message)


class CorrelationTestCase(unittest.TestCase):

    def test_correlation(self) -> None:
        u = np.random.normal(loc=0., scale=1., size=(100,))
        v = np.random.normal(loc=0., scale=1., size=(100,))
        r = utils.pearsonr(u, v)
        self.assertTrue(type(r) == np.float64)
        self.assertTrue(r >= float(-1) and r <= float(1))
