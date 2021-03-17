#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import torch
import utils

import numpy as np

from collections import defaultdict
from models.model import VSPoSE, SPoSE

def inference(
             modality:str,
             version:str,
             task:str,
             dim:int,
             lmbda:float,
             batch_size:int,
             results_dir:str,
             triplets_dir:str,
             device:torch.device,
             ) -> None:

    PATH = os.path.join(results_dir, modality, version, 'human', f'{dim}d', str(lmbda))
    rnd_seeds = [dir for dir in os.listdir(PATH) if dir[-2:].isdigit()]
    rnd_seeds = sorted(list(map(lambda str:int(str.replace('seed', '')), rnd_seeds)))
    test_accs = np.zeros(len(rnd_seeds))

    N_ITEMS = 1854
    #load validation set for hyperparam tuning
    tuning = False
    if tuning:
        _, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=False)
    else:
        test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=True)
    n_items = torch.max(test_triplets).item()
    if torch.min(test_triplets).item() == 0:
        n_items += 1
    I = torch.eye(n_items)
    test_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, I=I, batch_size=batch_size, inference=True)
    print(f'\nNumber of test batches in current process: {len(test_batches)}\n')

    avg_test_accs = np.zeros(len(rnd_seeds))
    model_confidence_scores = np.zeros((len(rnd_seeds), 11))
    avg_probability_scores = np.zeros((len(rnd_seeds), 11))
    avg_model_pmfs = defaultdict(list)

    for i, rnd_seed in enumerate(rnd_seeds):

        if version == 'variational':
            model = VSPoSE(in_size=N_ITEMS, out_size=dim, init_weights=True, init_method='normal', device=device, rnd_seed=rnd_seed)
        else:
            model = SPoSE(in_size=N_ITEMS, out_size=dim)
        try:
            model = utils.load_model(
                                    model=model,
                                    results_dir=results_dir,
                                    modality=modality,
                                    version=version,
                                    data='human',
                                    dim=dim,
                                    lmbda=lmbda,
                                    rnd_seed=rnd_seed,
                                    device=device,
            )
        except RuntimeError:
            raise Exception(f'\nModel parameters were incorrectly stored for random seed: {rnd_seed:02d}\n')

        test_acc, probas, model_pmfs = utils.test(
                                                    model=model,
                                                    test_batches=test_batches,
                                                    version=version,
                                                    task=task,
                                                    device=device,
                                                    batch_size=batch_size,
                                                    n_samples=20,
                                                    )
        avg_test_accs[i] += test_acc
        for triplet, pmf in model_pmfs.items():
            avg_model_pmfs[triplet].append(pmf)

        PATH = os.path.join(results_dir, modality, version, 'human', f'{dim}d', str(lmbda), f'seed{rnd_seed:02d}')
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        f_name = 'val_probas.npy' if tuning else 'test_probas.npy'
        with open(os.path.join(PATH, f_name), 'wb') as f:
            np.save(f, probas)

    avg_test_acc = np.mean(avg_test_accs)
    print(f'\nMean accuracy on held-out test set: {avg_test_acc}\n')

    avg_model_pmfs = {triplet: list(np.array(pmfs).mean(axis=0)) for triplet, pmfs in avg_model_pmfs.items()}

    PATH = os.path.join(results_dir, modality, version, 'human', f'{dim}d', str(lmbda), 'evaluation_metrics')
    PATH = os.path.join(PATH, 'validation') if tuning else PATH
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    utils.pickle_file(avg_model_pmfs, PATH, 'model_choice_pmfs')

if __name__ == '__main__':
    #parse os argument variables
    modality = sys.argv[1]
    version = sys.argv[2]
    task = sys.argv[3]
    dim = int(sys.argv[4])
    lmbda = float(sys.argv[5])
    batch_size = int(sys.argv[6])
    results_dir = sys.argv[7]
    triplets_dir = sys.argv[8]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference(
              modality=modality,
              version=version,
              task=task,
              dim=dim,
              lmbda=lmbda,
              batch_size=batch_size,
              results_dir=results_dir,
              triplets_dir=triplets_dir,
              device=device,
              )
