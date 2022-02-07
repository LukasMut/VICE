#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import torch
import unittest
import utils

import numpy as np
import helper

from typing import Dict, Iterator, List


batch_size = 128
test_dir = './test'
train_file = 'train_90.npy'
test_file = 'test_10.npy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TripletLoadingTestCase(unittest.TestCase):

    @staticmethod
    def save_triplets(train_triplets: np.ndarray, test_triplets: np.ndarray) -> None:
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        with open(os.path.join(test_dir, train_file), 'wb') as f:
            np.save(f, train_triplets)
        with open(os.path.join(test_dir, test_file), 'wb') as f:
            np.save(f, test_triplets)


    @staticmethod
    def vstack_items(item_order: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {i: torch.cat(items, dim=0) for i, items in item_order.items()}


    @staticmethod
    def assert_difference(A: torch.Tensor, B: torch.Tensor) -> bool:
        try:
            np.testing.assert_allclose(A.numpy(), B.numpy())
            return False
        except AssertionError:
            return True


    @staticmethod
    def collect_order(batches: Iterator[torch.Tensor]) ->  Dict[str, List[torch.Tensor]]:
        return {f'order_{order+1:02d}': [torch.nonzero(batch)[:, 1] for batch in batches] for order in range(2)}


    def test_batches(self) -> None:
        triplets = helper.create_triplets()
        train_triplets, test_triplets = helper.create_train_test_split(triplets)
        self.save_triplets(train_triplets, test_triplets)
        train_triplets, test_triplets = utils.load_data(device=device, triplets_dir=test_dir)
        shutil.rmtree(test_dir)

        self.assertEqual(train_triplets.shape[1], test_triplets.shape[1])
        self.assertEqual(type(train_triplets), type(test_triplets))
        self.assertTrue(isinstance(train_triplets, torch.Tensor))
        self.assertTrue(isinstance(test_triplets, torch.Tensor))

        M = utils.get_nitems(train_triplets)
        self.assertTrue(type(M) == int)

        train_batches, val_batches = utils.load_batches(
            train_triplets=train_triplets, test_triplets=test_triplets, n_items=M, batch_size=batch_size, inference=False)

        for i, batches in enumerate([train_batches, val_batches]):
            item_order = self.collect_order(batches)
            item_order = self.vstack_items(item_order)
            item_order_i = item_order['order_01']
            item_order_j = item_order['order_02']
            if i == 0:
                self.assertTrue(self.assert_difference(item_order_i, item_order_j))
            else:
                np.testing.assert_allclose(item_order_i.numpy(), item_order_j.numpy())
        