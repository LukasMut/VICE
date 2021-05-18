#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import numpy as np

from typing import List

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--triplets_dir', type=str,
        help='triplets directory (parent directory for full triplet dataset)')
    aa('--n_folds', type=int, default=10,
        choices=[40, 20, 10, 5],
        help='split data into k number of folds')
    aa('--additional_fractions', type=int, nargs='+',
        help='concatenate data folds to additionally create splits of (e.g., 10, 20, 50) % of full dataset')
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def split_data(X:np.ndarray, n_folds:int, out_path:str, file_name:str) -> None:
    X =  X[np.random.permutation(X.shape[0])]
    batch_size = int(X.shape[0]/n_folds)
    for i in range(n_folds):
        X_split = X[i*batch_size:(i+1)*batch_size]
        path = os.path.join(out_path, 'test', f'{int(100/n_folds):02d}', f'split_{i+1:02d}')
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, file_name), 'wb') as f:
            np.save(f, X_split)

def concatenate_(path:str, file_name:str, splits:List, p:int, k:int) -> None:
    X_train = np.vstack([np.load(os.path.join(split, file_name)) for split in splits])
    path = os.path.join(path, 'test', f'{p:02d}', f'split_{k+1:02d}')
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), 'wb') as f:
        np.save(f, X_train)

def merge_splits(parent_path:str, splits:List[str], fractions:List[int], file_name:str) -> None:
    for p in fractions:
        k = 0
        step_size = int(p/(100/len(splits)))
        for i in range(step_size, len(splits)+step_size, step_size):
            concatenate_(path=parent_path, file_name=file_name, splits=splits[i-step_size:i], p=p, k=k)
            k += 1

def get_splits(main_path:str, p:int=10) -> List[str]:
    split_path = os.path.join(main_path, 'test', f'{p:02d}')
    return sorted([os.path.join(split_path, d.name) for d in os.scandir(split_path) if d.is_dir() and d.name[-2:].isdigit()])

def create_splits(triplets_dir:str, n_folds:int, fractions:List[int]) -> None:
    file_name = 'train_90.npy'
    X_train = np.loadtxt(os.path.join(triplets_dir, 'train_90.txt'))
    split_data(X_train, n_folds, triplets_dir, file_name)
    splits = get_splits(triplets_dir, p=100//n_folds)
    merge_splits(triplets_dir, splits, fractions, file_name)

if __name__ == '__main__':
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    create_splits(
                    triplets_dir=args.triplets_dir,
                    n_folds=args.n_folds,
                    fractions=args.additional_fractions,
                    )
