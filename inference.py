#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from sklearn import mixture
from trainer import Trainer
from typing import Tuple, List, Any, Iterator
from models.model import VICE
from collections import defaultdict
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import re
import random

import torch
import utils
import itertools
import copy

os.environ['PYTHONIOENCODING'] = 'UTF-8'


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_items', type=int, default=1854,
        help='number of unique items/objects in dataset')
    aa('--latent_dim', type=int, default=100,
        help='initial dimensionality of VICE latent space')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--prior', type=str, metavar='p', default='gaussian',
        choices=['gaussian', 'laplace'],
        help='whether to use a mixture of Gaussians or Laplacians for the spike-and-slab prior')
    aa('--mc_samples', type=int,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of samples to use for MC sampling at inference time')
    aa('--n_components', type=int, nargs='+', default=None,
        help='number of clusters/modes in Gaussian mixture model')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir_test', type=str,
        help='directory from where to load triplets data')
    aa('--triplets_dir_val', type=str,
        help='directory from where to load triplets data')
    aa('--human_pmfs_dir', type=str, default=None,
        help='directory from where to load human choice probability distributions')
    aa('--pruning', action='store_true',
        help='whether model weights should be pruned prior to performing inference')
    aa('--things', action='store_true',
        help='whether pruning should be performed for models that were training on the THINGS objects')
    aa('--index_path', type=str, default=None,
        help='path/to/sortindex (sortindex is necessary to re-sort THINGS objects in the correct order')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args


def get_model_paths(PATH: str) -> List[str]:
    model_paths = []
    for root, _, files in os.walk(PATH):
        for f in files:
            if re.search(r'(?=^model)(?=.*epoch)(?=.*tar$)', f):
                model_paths.append('/'.join(root.split('/')[:-1]))
    return model_paths


def entropy(p: np.ndarray) -> np.ndarray:
    return np.sum(np.where(p == 0, 0, p * np.log(p)))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(q))


def kld(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return entropy(p) + cross_entropy(p, q)


def l1_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p - q, ord=1)


def compute_divergences(human_pmfs: dict, model_pmfs: dict, metric: str) -> np.ndarray:
    assert len(human_pmfs) == len(
        model_pmfs), '\nNumber of triplets in human and model distributions must correspond.\n'
    divergences = np.zeros(len(model_pmfs))
    for i, (triplet, p) in enumerate(human_pmfs.items()):
        q = np.asarray(model_pmfs[triplet])
        if metric == 'kld':
            div = kld(p, q)
        elif metric == 'cross-entropy':
            div = cross_entropy(p, q)
        else:
            div = l1_distance(p, q)
        divergences[i] += div
    return divergences


def prune_weights(model: Any, indices: torch.Tensor) -> Any:
    for m in model.parameters():
        m.data = m.data[indices]
    return model


def fit_gmm(X: np.ndarray, n_components: List[int]) -> Tuple[int, Any]:
    gmms, scores = [], []
    cv_types = ['spherical', 'diag', 'full']
    hyper_combs = list(itertools.product(cv_types, n_components))
    for comb in hyper_combs:
        gmm = mixture.GaussianMixture(
            n_components=comb[1], covariance_type=comb[0])
        gmm.fit(X)
        gmms.append(gmm)
        scores.append(gmm.bic(X))
    clusters = gmms[np.argmin(scores)].predict(X)
    n_clusters = hyper_combs[np.argmin(scores)][1]
    return clusters, n_clusters


def compute_pvals(W_mu: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    return np.array([norm.cdf(0, W_mu[:, j], W_b[:, j]) for j in range(W_mu.shape[1])])


def fdr_correction(p_vals: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.array(list(map(lambda p: multipletests(p, alpha=alpha, method='fdr_bh')[0], p_vals)))


def get_importance(rejections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(list(map(sum, rejections)))[:, np.newaxis]


def get_cluster_combinations(n_clusters: int, c_min: int = 1):
    combinations = []
    for k in range(c_min, n_clusters):
        combinations.extend(list(itertools.combinations(range(n_clusters), k)))
    return combinations


def pruning_(
    trainer: object,
    pruning_batches: Iterator[torch.Tensor],
    n_components: List[int],
    device: torch.device,
    things: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, nn.Module, float]:
    model = trainer.model
    W_loc, W_scale = trainer.parameters
    if things:
        W_loc = W_loc[sortindex]
        W_scale = W_scale[sortindex]
    importance = get_importance(fdr_correction(compute_pvals(W_loc, W_scale)))
    clusters, n_clusters = fit_gmm(importance, n_components)
    if n_clusters > 1:
        cluster_combs = get_cluster_combinations(n_clusters)
        val_losses = np.zeros(len(cluster_combs))
        pruned_models = []
        for i, comb in enumerate(cluster_combs):
            print(f'Testing cluster subset: {i+1}\n')
            if len(comb) > 1:
                indices = np.hstack([np.where(clusters == k)[0] for k in comb])
            else:
                indices = np.where(clusters == comb[0])[0]
            indices = torch.from_numpy(indices).type(
                torch.LongTensor).to(device)
            model_copy = copy.deepcopy(model)
            model_pruned = prune_weights(model_copy, indices)
            trainer.model = model_pruned
            val_loss, _ = trainer.evaluate(pruning_batches)
            val_losses[i] += val_loss
            pruned_models.append(model_pruned)
        best_model = pruned_models[np.argmin(val_losses)]
    else:
        best_model = model
    return best_model


def get_models(model_paths: List[str], prior: str, n_items: int, latent_dim: int, device: torch.device,
               ) -> List[nn.Module]:
    models = []
    for model_path in model_paths:
        seed = model_path.split('/')[-1]
        model = VICE(prior=prior, in_size=n_items,
                     out_size=latent_dim, init_weights=True)
        try:
            model = utils.load_model(
                model=model, PATH=model_path, device=device)
            models.append((seed, model))
        except RuntimeError:
            raise Exception(
                f'\nModel parameters were incorrectly stored for: {model_path}\n')
    return models


def inference(
    task: str,
    n_items: int,
    latent_dim: int,
    batch_size: int,
    prior: str,
    mc_samples: int,
    n_components: List[int],
    results_dir: str,
    triplets_dir_test: str,
    triplets_dir_val: str,
    human_pmfs_dir: str,
    pruning: bool,
    device: torch.device,
    things: bool = True,
) -> None:

    in_path = os.path.join(results_dir, f'{latent_dim}d')
    model_paths = get_model_paths(in_path)
    seeds, models = zip(*get_models(model_paths, prior,
                        n_items, latent_dim, device))

    test_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir_test, inference=True)
    _, pruning_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir_val, val_set='pruning_set', inference=False)
    _, tuning_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir_val, val_set='tuning_set', inference=False)

    test_batches = utils.load_batches(
        train_triplets=None, test_triplets=test_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    pruning_batches = utils.load_batches(
        train_triplets=None, test_triplets=pruning_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    tuning_batches = utils.load_batches(
        train_triplets=None, test_triplets=tuning_triplets, n_items=n_items, batch_size=batch_size, inference=True)

    print(
        f'\nNumber of pruning batches in current process: {len(pruning_batches)}\n')
    print(
        f'\nNumber of tuning batches in current process: {len(tuning_batches)}\n')
    print(
        f'\nNumber of test batches in current process: {len(test_batches)}\n')

    val_losses = dict()
    test_accs = dict()
    test_losses = dict()
    model_choices = dict()
    model_pmfs_all = defaultdict(dict)

    for seed, model, model_path in zip(seeds, models, model_paths):
        trainer = Trainer(
            model=model,
            task=task,
            N=None,
            n_items=n_items,
            latent_dim=latent_dim,
            optim=None,
            eta=None,
            batch_size=batch_size,
            epochs=None,
            mc_samples=mc_samples,
            prior=prior,
            spike=None,
            slab=None,
            pi=None,
            steps=None,
            model_dir=None,
            results_dir=results_dir,
            device=device,
        )
        if pruning:
            model_pruned = pruning_(
                trainer=trainer, pruning_batches=pruning_batches, n_components=n_components,  device=device, things=things)
        trainer.model = model_pruned
        val_loss, _ = trainer.evaluate(tuning_batches)
        test_acc, test_loss, probas, model_pmfs, triplet_choices = trainer.inference(
            test_batches)
        val_losses[seed] = val_loss
        test_accs[seed] = test_acc
        test_losses[seed] = test_loss
        model_pmfs_all[seed] = model_pmfs
        model_choices[seed] = triplet_choices

        print(f'Test accuracy for current random seed: {test_acc}\n')

        with open(os.path.join(model_path, 'test_probas.npy'), 'wb') as f:
            np.save(f, probas)

    seeds, _ = zip(*sorted(val_losses.items(),
                   key=lambda kv: kv[1], reverse=False))
    median_model = seeds[int(len(seeds) // 2) - 1]
    test_accs_ = list(test_accs.values())
    avg_test_acc = np.mean(test_accs_)
    median_test_acc = np.median(test_accs_)
    max_test_acc = np.max(test_accs_)
    print(f'\nMean accuracy on held-out test set: {avg_test_acc}')
    print(f'Median accuracy on held-out test set: {median_test_acc}')
    print(f'Max accuracy on held-out test set: {max_test_acc}\n')

    out_path = os.path.join(in_path, 'evaluation_metrics')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # sort accuracies in ascending order and cross-entropy errors in descending order
    test_accs = dict(
        sorted(test_accs.items(), key=lambda kv: kv[1], reverse=False))
    test_losses = dict(
        sorted(test_losses.items(), key=lambda kv: kv[1], reverse=True))
    # NOTE: we leverage the model that is slightly better than the median model (since we have 20 random seeds, the median is the average between model 10 and 11)
    # median_model = list(test_losses.keys())[int(len(test_losses)//2)-1]
    median_model_pmfs = model_pmfs_all[median_model]
    median_model_choices = model_choices[median_model]

    utils.pickle_file(median_model_choices, out_path, 'triplet_choices')
    utils.pickle_file(median_model_pmfs, out_path, 'model_choice_pmfs')

    utils.pickle_file(test_accs, out_path, 'test_accuracies')
    utils.pickle_file(test_losses, out_path, 'test_losses')

    human_pmfs = utils.unpickle_file(human_pmfs_dir, 'human_choice_pmfs')

    klds = compute_divergences(human_pmfs, median_model_pmfs, metric='kld')
    cross_entropies = compute_divergences(
        human_pmfs, median_model_pmfs, metric='cross-entropy')
    l1_distances = compute_divergences(
        human_pmfs, median_model_pmfs, metric='l1-distance')

    np.savetxt(os.path.join(out_path, 'klds_median.txt'), klds)
    np.savetxt(os.path.join(
        out_path, 'cross_entropies_median.txt'), cross_entropies)
    np.savetxt(os.path.join(out_path, 'l1_distances_median.txt'), l1_distances)

    klds, cross_entropies, l1_distances = {}, {}, {}
    for seed, model_pmfs in model_pmfs_all.items():
        klds[seed] = np.mean(compute_divergences(
            human_pmfs, model_pmfs, metric='kld'))
        cross_entropies[seed] = np.mean(compute_divergences(
            human_pmfs, model_pmfs, metric='cross-entropy'))
        l1_distances[seed] = np.mean(compute_divergences(
            human_pmfs, model_pmfs, metric='l1-distance'))

    utils.pickle_file(klds, out_path, 'klds_all')
    utils.pickle_file(cross_entropies, out_path, 'cross_entropies_all')
    utils.pickle_file(cross_entropies, out_path, 'l1_distances_all')


if __name__ == '__main__':
    # parse arguments and set random seed
    args = parseargs()
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    if args.things:
        assert isinstance(
            args.index_path, str), '\nPath to sortindex is missing.\n'
        try:
            global sortindex
            sortindex = pd.read_table(args.index_path, header=None)[0].values
        except FileNotFoundError:
            raise Exception(
                '\nDownload sortindex file for THINGS objects and provide correct path.\n')
    device = torch.device(args.device)
    inference(
        task=args.task,
        n_items=args.n_items,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        prior=args.prior,
        mc_samples=args.mc_samples,
        n_components=args.n_components,
        results_dir=args.results_dir,
        triplets_dir_test=args.triplets_dir_test,
        triplets_dir_val=args.triplets_dir_val,
        human_pmfs_dir=args.human_pmfs_dir,
        pruning=args.pruning,
        device=device,
        things=args.things,
    )
