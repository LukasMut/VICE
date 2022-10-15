#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import torch
import unittest
import utils
from data import TripletData

import numpy as np
import helper

from typing import Dict, Iterator, List

Array = np.ndarray
Tensor = torch.Tensor
batch_size = 128
test_dir = './test'
train_file = 'train_90.npy'
test_file = 'test_10.npy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TripletLoadingTestCase(unittest.TestCase):

    @staticmethod
    def save_triplets(train_triplets: Array, val_triplets: Array) -> None:
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        with open(os.path.join(test_dir, train_file), 'wb') as f:
            np.save(f, train_triplets)
        with open(os.path.join(test_dir, test_file), 'wb') as f:
            np.save(f, val_triplets)

    @staticmethod
    def vstack_items(item_order: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        return {i: torch.cat(items, dim=0) for i, items in item_order.items()}

    @staticmethod
    def assert_difference(A: Tensor, B: Tensor) -> bool:
        try:
            np.testing.assert_allclose(A.numpy(), B.numpy())
            return False
        except AssertionError:
            return True

    @staticmethod
    def collect_order(batches: Iterator[Tensor]) -> Dict[str, List[Tensor]]:
        return {f'order_{order+1:02d}': [torch.nonzero(batch)[:, -1] for batch in batches] for order in range(2)}

    def test_batches(self) -> None:
        triplets = helper.create_triplets()
        train_triplets, val_triplets = helper.create_train_test_split(
            triplets)
        self.save_triplets(train_triplets, val_triplets)
        
        train_triplets, val_triplets = utils.load_data(
            device=device, triplets_dir=test_dir)
        
        shutil.rmtree(test_dir)
        M = utils.get_nobjects(train_triplets)
        self.assertTrue(type(M) == int)
    
        train_triplets = TripletData(
            triplets=train_triplets, n_objects=M,
        )
        val_triplets = TripletData(
            triplets=val_triplets, n_objects=M,
        )
        self.assertEqual(train_triplets.triplets.shape[1], val_triplets.triplets.shape[1])
        self.assertEqual(type(train_triplets.triplets), type(val_triplets.triplets))
        self.assertTrue(isinstance(train_triplets.triplets, Tensor))
        self.assertTrue(isinstance(val_triplets.triplets, Tensor))

        train_batches = helper.get_batches(
            triplets=train_triplets, batch_size=batch_size, train=True)
        val_batches = helper.get_batches(
            triplets=val_triplets, batch_size=batch_size, train=False)
            
        for i, batches in enumerate([train_batches, val_batches]):
            item_order = self.collect_order(batches)
            item_order = self.vstack_items(item_order)
            item_order_i = item_order['order_01']
            item_order_j = item_order['order_02']
            if i == 0:
                self.assertTrue(self.assert_difference(
                    item_order_i, item_order_j))
            else:
                np.testing.assert_allclose(
                    item_order_i.numpy(), item_order_j.numpy())
