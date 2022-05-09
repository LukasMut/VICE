#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from collections import defaultdict

import numpy as np
import argparse
import os
import re
import random
import model
import torch
import utils

os.environ['PYTHONIOENCODING'] = 'UTF-8'
os.environ['OMP_NUM_THREADS'] = '1'


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--n_objects', type=int, default=1854,
        help='number of unique items/objects in dataset')
    aa('--latent_dim', type=int, default=100,
        help='initial dimensionality of VICE latent space')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--prior', type=str, metavar='p', default='gaussian',
        choices=['gaussian', 'laplace'],
        help='whether to use a mixture of Gaussians or Laplacians for the spike-and-slab prior')
    aa('--mc_samples', type=int, default=25,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of samples to use for MC sampling at inference time')
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--triplets_dir', type=str,
        help='directory from where to load validation and held-out test set')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args


def get_model_paths(PATH: str) -> List[str]:
    regex = r'(?=^model)(?=.*epoch)(?=.*tar$)'
    model_paths = []
    for root, _, files in os.walk(PATH):
        files = sorted(list(filter(lambda f: re.search(regex, f), files)))
        if files:
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


def prune_weights(model: model.VICE, indices: np.ndarray) -> model.VICE:
    for p in model.parameters():
        p.data = p.data[torch.from_numpy(indices)]
    return model


def pruning(model: model.VICE, alpha: float = .05, k: int = 5) -> model.VICE:
    params = model.detached_params
    loc = params['loc']
    scale = params['scale']
    p_vals = utils.compute_pvals(loc, scale)
    rejections = utils.fdr_corrections(p_vals, alpha)
    importance = utils.get_importance(rejections).ravel()
    signal = np.where(importance > k)[0]
    pruned_model = prune_weights(model, signal)
    return pruned_model


def get_models(
    vice_paths: List[str],
    task: str,
    prior: str,
    n_objects: int,
    latent_dim: int,
    batch_size: int,
    mc_samples: int,
    results_dir: str,
    device: torch.device,
    k: int = 5,
) -> list:
    vice_instances = []
    for vice_path in vice_paths:
        seed = vice_path.split('/')[-1]
        vice = getattr(model, 'VICE')(
            task=task,
            n_train=None,
            n_objects=n_objects,
            latent_dim=latent_dim,
            optim=None,
            eta=None,
            batch_size=batch_size,
            burnin=None,
            ws=None,
            k=k,
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
            init_weights=True)
        try:
            vice = utils.load_model(
                model=vice, PATH=vice_path, device=device)
            vice_instances.append((seed, vice))
        except RuntimeError:
            raise Exception(
                f'\nVICE parameters were incorrectly stored for: {vice_path}\n')
    return vice_instances


def inference(
    task: str,
    n_objects: int,
    latent_dim: int,
    batch_size: int,
    prior: str,
    mc_samples: int,
    results_dir: str,
    triplets_dir: str,
    device: torch.device,
) -> None:

    in_path = os.path.join(results_dir, f'{latent_dim}d')
    vice_paths = get_model_paths(in_path)
    seeds, vice_models = zip(*get_models(vice_paths, task, prior, n_objects,
                                latent_dim, batch_size, mc_samples, results_dir, device))

    test_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir, inference=True)
    _, val_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir, inference=False)

    test_batches = utils.load_batches(
        train_triplets=None, test_triplets=test_triplets, n_objects=n_objects, batch_size=batch_size, inference=True)
    val_batches = utils.load_batches(
        train_triplets=None, test_triplets=val_triplets, n_objects=n_objects, batch_size=batch_size, inference=True)

    print(
        f'\nNumber of validation batches in current process: {len(val_batches)}\n')
    print(
        f'Number of test batches in current process: {len(test_batches)}\n')

    val_losses = {}
    test_accs = {}
    test_losses = {}
    vice_choices = {}
    vice_pmfs_all = defaultdict(dict)

    for seed, vice, vice_path in zip(seeds, vice_models, vice_paths):
        pruned_vice = pruning(vice)
        val_loss, _ = pruned_vice.evaluate(val_batches)
        test_acc, test_loss, probas, vice_pmfs, triplet_choices = pruned_vice.inference(
            test_batches)
        val_losses[seed] = val_loss
        test_accs[seed] = test_acc
        test_losses[seed] = test_loss
        vice_pmfs_all[seed] = vice_pmfs
        vice_choices[seed] = triplet_choices

        print(f'Test accuracy for current random seed: {test_acc}\n')

        with open(os.path.join(vice_path, 'test_probas.npy'), 'wb') as f:
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
    median_model_pmfs = vice_pmfs_all[median_model]
    median_model_choices = vice_choices[median_model]

    utils.pickle_file(median_model_choices, out_path, 'triplet_choices')
    utils.pickle_file(median_model_pmfs, out_path, 'model_choice_pmfs')

    utils.pickle_file(test_accs, out_path, 'test_accuracies')
    utils.pickle_file(test_losses, out_path, 'test_losses')

    human_pmfs = utils.unpickle_file(triplets_dir, 'human_choice_pmfs')

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
    for seed, vice_pmfs in vice_pmfs_all.items():
        klds[seed] = np.mean(compute_divergences(
            human_pmfs, vice_pmfs, metric='kld'))
        cross_entropies[seed] = np.mean(compute_divergences(
            human_pmfs, vice_pmfs, metric='cross-entropy'))
        l1_distances[seed] = np.mean(compute_divergences(
            human_pmfs, vice_pmfs, metric='l1-distance'))

    utils.pickle_file(klds, out_path, 'klds_all')
    utils.pickle_file(cross_entropies, out_path, 'cross_entropies_all')
    utils.pickle_file(cross_entropies, out_path, 'l1_distances_all')


if __name__ == '__main__':
    # parse arguments and set random seed
    args = parseargs()
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    device = torch.device(args.device)
    inference(
        task=args.task,
        n_objects=args.n_objects,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        prior=args.prior,
        mc_samples=args.mc_samples,
        results_dir=args.results_dir,
        triplets_dir=args.triplets_dir,
        device=device,
    )
