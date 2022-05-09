#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import h5py
import random
import re
import scipy.io
import numpy as np
import itertools

from typing import Tuple

os.environ["PYTHONIOENCODING"] = "UTF-8"


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--in_path", type=str, help="path/to/design/matrix")
    aa("--out_path", type=str, help="path/to/triplets")
    aa("--n_samples", type=int, help="number of triplet samples")
    aa("--rnd_seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    return args


class Tripletizer(object):
    def __init__(
        self,
        in_path: str,
        out_path: str,
        n_samples: int,
        rnd_seed: int,
        k: int = 3,
        train_frac: float = 0.8,
    ):
        self.in_path = in_path
        self.out_path = out_path
        self.n_samples = n_samples
        self.k = k
        self.train_frac = train_frac

        if not re.search(r"(mat|txt|csv|npy|hdf5)$", in_path):
            raise Exception(
                "\nCannot tripletize input data other than .mat, .txt, .csv, .npy, or .hdf5 formats\n"
            )

        if not os.path.exists(self.out_path):
            print(f"\n....Creating output directory: {self.out_path}\n")
            os.makedirs(self.out_path)

        np.random.seed(rnd_seed)
        random.seed(rnd_seed)

    def load_domain(self, in_path: str) -> np.ndarray:
        if re.search(r"mat$", in_path):
            X = np.vstack(
                [
                    v
                    for v in scipy.io.loadmat(in_path).values()
                    if isinstance(v, np.ndarray) and v.dtype == np.float
                ]
            )
        elif re.search(r"txt$", in_path):
            X = np.loadtxt(in_path)
        elif re.search(r"csv$", in_path):
            X = np.loadtxt(in_path, delimiter=",")
        elif re.search(r"npy$", in_path):
            X = np.load(in_path)
        elif re.search(r"hdf5$", in_path):
            with h5py.File(in_path, "r") as f:
                X = list(f.values())[0][:]
        else:
            raise Exception("\nInput data does not seem to be in the right format\n")
        X = self.remove_nans_(X)
        return X

    @staticmethod
    def remove_nans_(X: np.ndarray) -> np.ndarray:
        nan_indices = np.isnan(X[:, :]).any(axis=1)
        return X[~nan_indices]

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        return np.exp(z) / np.sum(np.exp(z))

    def get_choice(self, S: np.ndarray, triplet: np.ndarray) -> np.ndarray:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        # TODO: change temperature value (i.e., beta) because 
        # softmax yields NaNs if similarity values are too large
        # probas = self.softmax(sims)
        # positive = combs[np.argmax(probas)]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))
        return choice

    @staticmethod
    def random_choice(n_samples: int, combs: np.ndarray):
        return combs[np.random.choice(combs.shape[0], size=n_samples, replace=False)]

    def sample_triplets(self) -> np.ndarray:
        """Create synthetic triplet data."""
        X = self.load_domain(self.in_path)
        M = X.shape[0]
        S = X @ X.T
        triplets = np.zeros((self.n_samples, self.k), dtype=int)
        combs = np.array(list(itertools.combinations(range(M), self.k)))
        random_sample = self.random_choice(self.n_samples, combs)
        for i, triplet in enumerate(random_sample):
            choice = self.get_choice(S, triplet)
            triplets[i] = choice
        return triplets

    def create_train_test_split(
        self, triplets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split triplet data into train and test splits."""
        N = triplets.shape[0]
        rnd_perm = np.random.permutation(N)
        train_split = triplets[rnd_perm[: int(len(rnd_perm) * self.train_frac)]]
        test_split = triplets[rnd_perm[int(len(rnd_perm) * self.train_frac):]]
        return train_split, test_split

    def save_triplets(self, triplets: np.ndarray) -> None:
        train_split, test_split = self.create_train_test_split(triplets)
        with open(os.path.join(self.out_path, "train_90.npy"), "wb") as train_file:
            np.save(train_file, train_split)
        with open(os.path.join(self.out_path, "test_10.npy"), "wb") as test_file:
            np.save(test_file, test_split)


if __name__ == "__main__":
    # parse arguments
    args = parseargs()

    tripletizer = Tripletizer(
        in_path=args.in_path,
        out_path=args.out_path,
        n_samples=args.n_samples,
        rnd_seed=args.rnd_seed,
    )
    triplets = tripletizer.sample_triplets()
    tripletizer.save_triplets(triplets)
