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
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_items', type=int, default=1854,
        help='number of unique items/objects in dataset')
    aa('--dim', type=int, default=100,
        help='latent dimensionality of VICE representations')
    aa('--temperatures', type=float, nargs='+',
        help='temperature values for scaling the embeddings')
    aa('--n_samples', type=int,
        choices=[10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of weight samples used in Monte Carlo (MC) sampling')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets data')
    aa('--compute_stds', action='store_true',
        help='whether to compute standard deviations of predicted probabilities')
    aa('--device', type=str,
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'])
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
             process_id:int,
             modality:str,
             task:str,
             n_items:int,
             dim:int,
             temperatures:List[float],
             n_samples:int,
             batch_size:int,
             results_dir:str,
             triplets_dir:str,
             device:torch.device,
             compute_stds:bool=False,
             ) -> None:

    PATH = os.path.join(results_dir, modality, 'variational', f'{dim}d')
    model_paths = get_model_paths(PATH)
    temp = temperatures[process_id]
    _, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=False)
    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    val_centropies = dict()
    val_accs = dict()

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
            val_acc, val_loss, probas, model_pmfs, stds = utils.test(
                                                                    model=model,
                                                                    test_batches=test_batches,
                                                                    task=task,
                                                                    batch_size=batch_size,
                                                                    n_samples=n_samples,
                                                                    device=device,
                                                                    temp=temp,
                                                                    compute_stds=compute_stds,
                                                                    )
            triplet_stds[seed] = stds
        else:
            val_acc, val_loss, probas, model_pmfs = utils.test(
                                                              model=model,
                                                              test_batches=test_batches,
                                                              task=task,
                                                              batch_size=batch_size,
                                                              n_samples=n_samples,
                                                              device=device,
                                                              temp=temp,
                                                              )

        val_centropies[seed] = val_loss
        val_accs[seed] = val_acc

        print(f'Validation accuracy for current random seed: {val_acc}')
        print(f'Validation cross-entropy for current random seed: {val_loss}\n')

        f_name = 'val_probas.npy'
        with open(os.path.join(model_path, f_name), 'wb') as f:
            np.save(f, probas)

    val_accs_ = list(val_accs.values())
    avg_val_acc = np.mean(val_accs_)
    median_val_acc = np.median(val_accs_)
    max_val_acc = np.max(val_accs_)
    print(f'\nMean accuracy on validation set: {avg_val_acc}')
    print(f'Median accuracy on validation set: {median_val_acc}')
    print(f'Max accuracy on validation set: {max_val_acc}\n')

    PATH = os.path.join(PATH, 'evaluation_metrics', 'validation', f'{temp:.2f}')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    utils.pickle_file(val_accs, PATH, 'val_accs')
    utils.pickle_file(val_centropies, PATH, 'val_centropies')

    if compute_stds:
        utils.pickle_file(triplet_stds, PATH, 'triplet_stds')

if __name__ == '__main__':
    args = parseargs()
    n_procs = len(args.temperatures)
    torch.multiprocessing.set_start_method('spawn', force=True)

    if re.search(r'^cuda', args.device):
        try:
            current_device = int(args.device[-1])
        except ValueError:
            current_device = 1
        try:
            torch.cuda.set_device(current_device)
        except RuntimeError:
            torch.cuda.set_device(0)
        print(f'\nPyTorch CUDA version: {torch.version.cuda}')
    else:
        if n_procs > os.cpu_count()-1:
            raise Exception(f'CPU node cannot run {n_procs} in parallel. Maximum number of processes is {os.cpu_count()-1}.\n')

    print(f'\nRunning {n_procs} processes in parallel.\n')
    torch.multiprocessing.spawn(
                                inference,
                                args=(
                                args.modality,
                                args.task,
                                args.n_items,
                                args.dim,
                                args.temperatures,
                                args.n_samples,
                                args.batch_size,
                                args.results_dir,
                                args.triplets_dir,
                                args.device,
                                args.compute_stds,
                                ),
                                nprocs=n_procs,
                                join=True)
