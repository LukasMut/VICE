#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch

from dataclasses import dataclass
from typing import Any, Iterator

Tensor = Any
@dataclass
class DataLoader:
    dataset: Tensor
    n_objects: int
    batch_size: int
    train: bool = True
    
    def __post_init__(self) -> None:
        # initialize an identity matrix of size m x m for one-hot-encoding of triplets
        self.identity = torch.eye(self.n_objects)
        self.n_batches = int(math.ceil(len(self.dataset) / self.batch_size))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[Tensor]:
        return self.get_batches(self.dataset)

    def get_batches(self, triplets: Tensor) -> Iterator[Tensor]:
        if self.train:
            triplets = triplets[torch.randperm(triplets.shape[0])]
        for i in range(self.n_batches):
            batch = self.encode_as_onehot(
                triplets[i * self.batch_size: (i + 1) * self.batch_size]
            )
            yield batch

    def encode_as_onehot(self, triplets: Tensor) -> Tensor:
        """encode item triplets as one-hot-vectors"""
        return self.identity[triplets.flatten(), :]
