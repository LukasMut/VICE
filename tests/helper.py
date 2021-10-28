#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import itertools

from typing import Tuple


N_TRIALS = int(1e+4)
N_ITEMS = 60
LATENT_DIM = N_ITEMS // 2
k = 3

def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z))


def get_choice(S: np.ndarray, triplet: np.ndarray) -> np.ndarray:
    combs = list(itertools.combinations(triplet, 2))
    sims = [S[comb[0], comb[1]] for comb in combs]
    probas = softmax(sims)
    positive = combs[np.argmax(probas)]
    ooo = list(set(triplet).difference(set(positive)))
    choice = np.hstack((positive, ooo))
    return choice.tolist()


def random_choice(N: int, combs: np.ndarray):
    random_sample = np.random.choice(combs.shape[0], size=N, replace=False)
    return combs[random_sample]


def create_triplets(N: int=N_TRIALS, M: int=N_ITEMS, P: int=LATENT_DIM, k: int=k) -> np.ndarray:
    """Create synthetic triplet data."""
    X = np.random.randn(M, P)
    S = X @ X.T
    triplets = torch.zeros(N, k, dtype=torch.long)
    combs = np.array(list(itertools.combinations(range(M), k)))
    random_sample = random_choice(N, combs)
    for i, triplet in enumerate(random_sample):
        choice = get_choice(S, triplet)
        triplets[i] = choice
    return triplets


def create_train_test_split(triplets: np.ndarray, train_frac: float=.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split triplet data into train and test splits."""
    N = triplets.shape[0]
    rnd_perm = np.random.permutation(N)
    train_split = triplets[rnd_perm[:int(len(rnd_perm) * train_frac)]]
    test_split = triplets[rnd_perm[int(len(rnd_perm) * train_frac):]]
    return train_split, test_split
