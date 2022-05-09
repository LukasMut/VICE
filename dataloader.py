#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch

from typing import Iterator


class DataLoader(object):
    def __init__(
        self,
        dataset: torch.Tensor,
        n_objects: int,
        batch_size: int,
        train: bool = True,
    ):
        self.dataset = dataset
        # initialize an identity matrix of size m x m for one-hot-encoding of triplets
        self.identity = torch.eye(n_objects)
        self.batch_size = batch_size
        self.train = train
        self.n_batches = int(math.ceil(len(self.dataset) / self.batch_size))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_batches(self.dataset)

    def get_batches(self, triplets: torch.Tensor) -> Iterator[torch.Tensor]:
        if self.train:
            triplets = triplets[torch.randperm(triplets.shape[0])]
        for i in range(self.n_batches):
            batch = self.encode_as_onehot(
                triplets[i * self.batch_size: (i + 1) * self.batch_size]
            )
            yield batch

    def encode_as_onehot(self, triplets: torch.Tensor) -> torch.Tensor:
        """encode item triplets as one-hot-vectors"""
        return self.identity[triplets.flatten(), :]
