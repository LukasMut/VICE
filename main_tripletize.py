#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import h5py
import random
import math
import re
import scipy.io
import numpy as np
import itertools
from collections import Counter
from typing import Any, Tuple
from dataclasses import dataclass

Array = Any
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


@dataclass
class Tripletizer:
    in_path: str
    out_path: str
    n_samples: int
    rnd_seed: int
    k: int = 3
    train_frac: float = 0.9

    def __post_init__(self) -> None:
        if not re.search(r"(mat|txt|csv|npy|hdf5)$", self.in_path):
            raise Exception(
                "\nCannot tripletize input data other than .mat, .txt, .csv, .npy, or .hdf5 formats\n"
            )
        if not os.path.exists(self.out_path):
            print(f"\n....Creating output directory: {self.out_path}\n")
            os.makedirs(self.out_path)

        random.seed(self.rnd_seed)
        np.random.seed(self.rnd_seed)

    def load_domain(self, in_path: str) -> Array:
        if re.search(r"mat$", in_path):
            X = np.vstack(
                [
                    v
                    for v in scipy.io.loadmat(in_path).values()
                    if isinstance(v, Array) and v.dtype == np.float
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
    def remove_nans_(X: Array) -> Array:
        nan_indices = np.isnan(X[:, :]).any(axis=1)
        return X[~nan_indices]

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        proba = np.exp(z) / np.sum(np.exp(z))
        return proba 

    @staticmethod
    def log_softmax_scaled(z: np.ndarray, const) -> np.ndarray:
        ''' see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/'''  
        z = z - const
        scaled_proba = np.exp(z) / np.sum(np.exp(z))
        scaled_log_proba = const + np.log(scaled_proba)
        return scaled_log_proba

    def get_choice(self, S: Array, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        # TODO: change temperature value (i.e., beta param) because 
        # softmax yields NaNs if dot products are too large
        # probas = self.softmax(sims)
        # positive = combs[np.argmax(probas)]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))
        return choice

    @staticmethod
    def random_choice(n_samples: int, combs: Array):
        return combs[np.random.choice(combs.shape[0], size=n_samples, replace=False)]

    def sample_triplets(self) -> Array:
        """Create synthetic triplet data."""
        X = self.load_domain(self.in_path)
        M = X.shape[0]

        # This is a matrix of N x N (i.e. image_features x image_features). 
        # i.e. The dot product between the corresponding network representations
        S = X @ X.T 
        
        # Adaptive sampling of unique triplets
        unique_triplets = set()
        count = Counter()
        count.update({x:0 for x in range(M)})

        # At the start all classes have zero counts and we sample uniformly
        p_per_item = [1 / M for _ in range(M)] 
        sample_idx, n_iter = 1, 1
        while sample_idx < self.n_samples+1:
            n_iter += 1
            print(f'{n_iter} samples drawn, {sample_idx}/{self.n_samples} added', end='\r')
            triplet = np.random.choice(range(M), 3, replace=False, p=p_per_item)

            # Using this we can avoid duplicate triplets when adding to the set
            triplet.sort() 
            triplet = tuple(triplet)

            # Add to set and increase count if triplet is still unique
            if triplet not in unique_triplets:
                count.update(triplet)
                unique_triplets.add(triplet)
                sample_idx += 1 

            # Update histogram of each class and sample random choices with the inverse of the actual distribution
            if sample_idx % 100000 == 0:
                sum_count = sum(count.values())
                sorted_count = sorted(count.items())

                # Make smallest proba the largest
                inverse_probas_per_item = [1 - s[1] / sum_count for s in sorted_count] 
                
                # Correct uniform distribution
                norm_probas =  [float(i)/sum(inverse_probas_per_item) for i in inverse_probas_per_item] 
                p_per_item = norm_probas


        triplets = np.zeros((self.n_samples, self.k), dtype=int)
        for i, triplet in enumerate(unique_triplets):
            print(f'Process {i}/{self.n_samples} triplets', end='\r')
            choice = self.get_choice(S, triplet)
            triplets[i] = choice # probably returns a list of indices of shape k where for that image the odd one out is

        return triplets

    def create_train_test_split(
        self, triplets: Array
    ) -> Tuple[Array, Array]:
        """Split triplet data into train and test splits."""
        N = triplets.shape[0]
        rnd_perm = np.random.permutation(N)
        train_split = triplets[rnd_perm[: int(len(rnd_perm) * self.train_frac)]]
        test_split = triplets[rnd_perm[int(len(rnd_perm) * self.train_frac):]]
        return train_split, test_split

    def save_triplets(self, triplets: Array) -> None:
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
