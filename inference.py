#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
import torch
import utils

import numpy as np

from collections import defaultdict
from models.model import VSPoSE
from typing import List, Dict

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--modality', type=str,
        help='current modality (e.g., behavioral, synthetic)')
    aa('--task', type=str,
        choices=['odd_one_out', 'similarity_task'])
    aa('--dim', type=int,
        help='latent dimensionality of VSPoSE embedding matrices')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets data')
    aa('--tuning', action='store_true',
        help='whether to run script for tuning alpha (smoothing parameter)')
    args = parser.parse_args()
    return args

def get_model_paths(PATH:str) -> List[str]:
    model_paths = []
    for seed in os.scandir(PATH):
        if seed.is_dir() and seed.name[-2:].isdigit():
            seed_path = os.path.join(PATH, seed.name)
            for root, dirs, files in os.walk(seed_path):
                 for name in files:
                      if name.endswith('.json'):
                          model_paths.append(root)
    return model_paths

def inference(
             modality:str,
             task:str,
             dim:int,
             batch_size:int,
             results_dir:str,
             triplets_dir:str,
             device:torch.device,
             tuning:bool=False,
             ) -> None:

    PATH = os.path.join(results_dir, modality, 'variational', f'{dim}d')
    model_paths = get_model_paths(PATH)
    N_ITEMS = 1854
    #load validation set for hyperparam tuning
    if tuning:
        _, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=False)
    else:
        test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=True)

    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=N_ITEMS, batch_size=batch_size, inference=True)
    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    test_accs = dict()
    model_pmfs_all = defaultdict(dict)

    for model_path in model_paths:
        model = VSPoSE(in_size=N_ITEMS, out_size=dim, init_weights=True, init_method='normal', device=device)
        try:
            model = utils.load_model(model=model, PATH=model_path, device=device)
        except RuntimeError:
            raise Exception(f'\nModel parameters were incorrectly stored for: {model_path}\n')

        test_acc, probas, model_pmfs = utils.test(model=model, test_batches=test_batches, task=task, batch_size=batch_size, n_samples=20, device=device)
        seed = model_path.split('/')[-3]
        test_accs[seed] = test_acc
        model_pmfs_all[seed] = model_pmfs

        print(f'Test accuracy for current random seed: {test_acc}')

        f_name = 'val_probas.npy' if tuning else 'test_probas.npy'
        with open(os.path.join(model_path, f_name), 'wb') as f:
            np.save(f, probas)

    avg_test_acc = np.mean(list(test_accs.values()))
    median_test_acc = np.median(list(test_accs.values()))
    max_test_acc = np.max(list(test_accs.values()))
    print(f'\nMean accuracy on held-out test set: {avg_test_acc}')
    print(f'Median accuracy on held-out test set: {median_test_acc}')
    print(f'Max accuracy on held-out test set: {max_test_acc}\n')

    PATH = os.path.join(PATH, 'evaluation_metrics')
    PATH = os.path.join(PATH, 'validation') if tuning else PATH
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    utils.pickle_file(model_pmfs_all, PATH, 'model_choice_pmfs')

    if not tuning:
        with open(os.path.join(PATH, 'test_accuracies.npy'), 'wb') as f:
            np.save(f, test_accs)

if __name__ == '__main__':
    args = parseargs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference(
              modality=args.modality,
              task=args.task,
              dim=args.dim,
              batch_size=args.batch_size,
              results_dir=args.results_dir,
              triplets_dir=args.triplets_dir,
              device=device,
              tuning=args.tuning,
              )
