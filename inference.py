#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
import torch
import utils

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from models.model import VICE
from typing import List, Dict

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--modality', type=str,
        help='current modality (e.g., behavioral, synthetic)')
    aa('--task', type=str,
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_items', type=int, default=1854,
        help='number of unique items/objects in dataset')
    aa('--dim', type=int, default=100,
        help='latent dimensionality of VICE representations')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--n_samples', type=int,
        choices=[10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of samples to use for MC sampling at inference time')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets data')
    aa('--human_pmfs_dir', type=str, default=None,
        help='directory from where to load human choice probability distributions')
    aa('--compute_stds', action='store_true',
        help='whether to compute standard deviations of predicted probabilities')
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

def smoothing_(p:np.ndarray, alpha:float=.1) -> np.ndarray:
    return (p + alpha) / np.sum(p + alpha)

def entropy_(p:np.ndarray) -> np.ndarray:
    return np.sum(np.where(p == 0, 0, p*np.log(p)))

def cross_entropy_(p:np.ndarray, q:np.ndarray, alpha:float) -> float:
    return -np.sum(p*np.log(smoothing_(q, alpha)))

def kld_(p:np.ndarray, q:np.ndarray, alpha:float) -> float:
    return entropy_(p) + cross_entropy_(p, q, alpha)

def compute_divergences(human_pmfs:dict, model_pmfs:dict, alpha:float, metric:str='kld'):
    assert len(human_pmfs) == len(model_pmfs), '\nNumber of triplets in human and model distributions must correspond.\n'
    divergences = np.zeros(len(model_pmfs))
    accuracy = 0
    for i, (triplet, p) in enumerate(human_pmfs.items()):
        q = np.asarray(model_pmfs[triplet])
        div = kld_(p, q, alpha) if metric  == 'kld' else cross_entropy_(p, q, alpha)
        divergences[i] += div
    return divergences

def get_val_pmfs(triplets_dir:str) -> dict:
    val_set = np.loadtxt(os.path.join(triplets_dir, 'test_10.txt'))
    val_pmfs = {tuple(np.sort(t).astype(int)): np.array([0, 0, 1])[np.argsort(t)] for t in val_set}
    return val_pmfs

def get_triplet_intersection(val_pmfs:dict, seed_pmfs:dict) -> dict:
    #this is necessary since we perform batch-wise inference which might skip the last batch (due to "n_triplets % batch_size != 0")
    intersection = set(list(val_pmfs.keys())).intersection(set(list(seed_pmfs.keys())))
    val_pmfs = {t:val_pmfs[t] for t in intersection}
    return val_pmfs

def laplace_smoothing(triplets_dir:str, model_pmfs:dict, smoothing_values:np.ndarray) -> dict:
    alphas = {}
    for seed, seed_pmfs in model_pmfs.items():
        entropies = np.zeros(len(smoothing_values))
        val_pmfs = get_val_pmfs(triplets_dir)
        val_pmfs = get_triplet_intersection(val_pmfs, seed_pmfs)
        for k, alpha in enumerate(smoothing_values):
            cross_entropies = compute_divergences(val_pmfs, seed_pmfs, alpha, metric='cross-entropy')
            entropies[k] += np.mean(cross_entropies)
        alphas[seed] = smoothing_values[np.argmin(entropies)]
    return alphas

def inference(
             modality:str,
             task:str,
             n_items:int,
             dim:int,
             batch_size:int,
             n_samples:int,
             results_dir:str,
             triplets_dir:str,
             human_pmfs_dir:str,
             device:torch.device,
             compute_stds:bool=False,
             ) -> None:

    PATH = os.path.join(results_dir, modality, 'variational', f'{dim}d')
    model_paths = get_model_paths(PATH)
    test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=True)
    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    test_accs = dict()
    test_losses = dict()
    model_pmfs_all = defaultdict(dict)

    if compute_stds:
        triplet_stds = dict()

    for model_path in model_paths:
        seed = model_path.split('/')[-4]
        model = VICE(in_size=n_items, out_size=dim, init_weights=True)
        try:
            model = utils.load_model(model=model, PATH=model_path, device=device)
        except RuntimeError:
            raise Exception(f'\nModel parameters were incorrectly stored for: {model_path}\n')

        if compute_stds:
            test_acc, test_loss, probas, model_pmfs, stds = utils.test(
                                                                        model=model,
                                                                        test_batches=test_batches,
                                                                        task=task,
                                                                        batch_size=batch_size,
                                                                        n_samples=n_samples,
                                                                        device=device,
                                                                        compute_stds=compute_stds,
                                                                        )
            triplet_stds[seed] = stds
        else:
            test_acc, test_loss, probas, model_pmfs = utils.test(
                                                                  model=model,
                                                                  test_batches=test_batches,
                                                                  task=task,
                                                                  batch_size=batch_size,
                                                                  n_samples=n_samples,
                                                                  device=device,
                                                                  )

        test_accs[seed] = test_acc
        test_losses[seed] = test_loss
        model_pmfs_all[seed] = model_pmfs

        print(f'Test accuracy for current random seed: {test_acc}')

        with open(os.path.join(model_path, 'test_probas.npy'), 'wb') as f:
            np.save(f, probas)

    test_accs_ = list(test_accs.values())
    avg_test_acc = np.mean(test_accs_)
    median_test_acc = np.median(test_accs_)
    max_test_acc = np.max(test_accs_)
    print(f'\nMean accuracy on held-out test set: {avg_test_acc}')
    print(f'Median accuracy on held-out test set: {median_test_acc}')
    print(f'Max accuracy on held-out test set: {max_test_acc}\n')

    PATH = os.path.join(PATH, 'evaluation_metrics')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    utils.pickle_file(model_pmfs_all, PATH, 'model_choice_pmfs')
    utils.pickle_file(test_accs, PATH, 'test_accuracies')
    test_accs = dict(sorted(test_accs.items(), key=lambda kv:kv[1], reverse=True))
    #NOTE: we leverage the model that is slightly better than the median model (since we have 20 random seeds, the median is the average between model 10 and 11)
    best_model = list(test_accs.keys())[0]

    human_pmfs = utils.unpickle_file(human_pmfs_dir, 'human_choice_pmfs')
    best_model_pmfs = model_pmfs_all[best_model]
    alpha = 0

    klds = compute_divergences(human_pmfs, best_model_pmfs, alpha, metric='kld')
    cross_entropies = compute_divergences(human_pmfs, best_model_pmfs, alpha, metric='cross-entropy')

    np.savetxt(os.path.join(PATH, 'klds.txt'), klds)
    np.savetxt(os.path.join(PATH, 'cross_entropies.txt'), cross_entropies)

    avg_kld = np.mean(klds)
    avg_cross_entropy = np.mean(cross_entropies)

    print(f'Average KLD: {avg_kld}')
    print(f'Average cross-entropy: {avg_cross_entropy}\n')

    if compute_stds:
        utils.pickle_file(triplet_stds, PATH, 'triplet_stds')

if __name__ == '__main__':
    args = parseargs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(
              modality=args.modality,
              task=args.task,
              n_items=args.n_items,
              dim=args.dim,
              batch_size=args.batch_size,
              n_samples=args.n_samples,
              results_dir=args.results_dir,
              triplets_dir=args.triplets_dir,
              human_pmfs_dir=args.human_pmfs_dir,
              device=device,
              compute_stds=args.compute_stds,
              )
