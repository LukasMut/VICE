#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import numpy as np

from collections import defaultdict


def load_data(in_path):
    first_half = np.loadtxt(os.path.join(in_path, 'apns1.txt'), dtype=int)
    second_half = np.loadtxt(os.path.join(in_path, 'apns2.txt'), dtype=int)
    return first_half, second_half


def reorder_triplet(t):
    return tuple(sorted(t))


def get_repeats_and_unique_triplets(triplets):
    repeats = defaultdict(list)
    unique_triplets = []
    for t in triplets:
        t_sorted = reorder_triplet(t)
        if not t_sorted in unique_triplets:
            unique_triplets.append(t_sorted)
        repeats[t_sorted].append(list(t))
    return repeats, unique_triplets


def roll_dice(outcomes, probabilities):
    return np.random.choice(outcomes, replace=True, p=probabilities)


def partition_triplets(triplets, repeats, unique_triplets):
    partition_i = [] # test split
    partition_j = [] # train split
    partition_k = [] # val split
    outcomes = [0, 1, 2]
    probabilities = [0.5, 0.45, 0.05]
    for t in triplets:
        t_sorted = reorder_triplet(t)
        if t_sorted in unique_triplets:
            outcome = roll_dice(outcomes, probabilities)
            if outcome == 0:
                partition_i.extend(repeats[t_sorted])
            elif outcome == 1:
                partition_j.extend(repeats[t_sorted])
            else:
                partition_k.extend(repeats[t_sorted])
            idx = unique_triplets.index(t_sorted)
            unique_triplets.pop(idx)
    partition_i = np.random.permutation(partition_i)
    partition_j = np.random.permutation(partition_j)
    partition_k = np.random.permutation(partition_k)
    return partition_i, partition_j, partition_k


def save_partitions(partitions, out_path):
    file_names = ['test_triplets', 'train_90', 'test_10']
    for i, partition in enumerate(partitions):
        with open(os.path.join(out_path, f'{file_names[i]}.npy'), 'wb') as f:
                  np.save(f, partition)


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    rnd_seed = int(sys.argv[3])

    # seed rng
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)
    
    first_half, second_half = load_data(in_path)
    triplets = np.concatenate((first_half, second_half), axis=0)
    repeats, unique_triplets = get_repeats_and_unique_triplets(triplets)
    partitions = partition_triplets(triplets, repeats, unique_triplets)
    # save_partitions(partitions, out_path)

