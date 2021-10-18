#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import random
import re
import torch
import utils
import itertools
import copy

import numpy as np
import pandas as pd
import torch.nn as nn

from models.model import VICE
from trainer import Trainer
from os.path import join as pjoin
from typing import Tuple, List, Any, Iterator
from sklearn import mixture
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

os.environ['PYTHONIOENCODING'] = 'UTF-8'
# number of cores used per Python process (set to 2 if HT is enabled, else keep 1)
os.environ['OMP_NUM_THREADS'] = '1'


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--modality', type=str,
        help='current modality (e.g., behavioral, fMRI, text, DNNs)')
    aa('--n_items', type=int,
        help='number of unique items/stimuli/objects in dataset')
    aa('--latent_dim', type=int, default=100,
        help='latent dimensionality of VICE representations')
    aa('--thresh', type=float, default=0.8,
        choices=[0.75, 0.8, 0.85, 0.9, 0.95],
        help='reproducibility threshold (0.8 used in the ICLR paper)')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--optim', type=str, metavar='o', default='adam',
        choices=['adam', 'adamw', 'sgd'],
        help='optimizer that was used to train VICE')
    aa('--prior', type=str, metavar='p', default='gaussian',
        choices=['gaussian', 'laplace'],
        help='whether to use a mixture of Gaussians or Laplacians for the spike-and-slab prior')
    aa('--spike', type=float,
        help='sigma of spike distribution')
    aa('--slab', type=float,
        help='sigma of slab distribution')
    aa('--pi', type=float,
        help='probability value with which to sample from the spike')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets data')
    aa('--n_components', type=int, nargs='+',
        help='number of clusters/modes in the Gaussian Mixture Model (GMM) used for pruning')
    aa('--mc_samples', type=int,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of weight samples used in Monte Carlo (MC) sampling')
    aa('--things', action='store_true',
        help='whether pruning should be performed for models that were training on the THINGS objects')
    aa('--index_path', type=str, default=None,
        help='path/to/sortindex (sortindex is necessary to re-sort THINGS objects in the correct order')
    aa('--device', type=str,
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility of results')
    args = parser.parse_args()
    return args


def avg_ndims(Ws_mu: list) -> np.ndarray:
    return np.ceil(np.mean(list(map(lambda w: min(w.shape), Ws_mu))))


def std_ndims(Ws_mu: list) -> np.ndarray:
    return np.std(list(map(lambda w: min(w.shape), Ws_mu)))


def robustness(corrs: np.ndarray, thresh: float) -> float:
    return len(corrs[corrs > thresh]) / len(corrs)


def uncertainty_estimates(W_sigma: np.ndarray, sorted_dims: np.ndarray, rel_freq: float) -> float:
    W_sigma_mean = np.mean(W_sigma, axis=0)
    assert len(W_sigma_mean) == min(W_sigma.shape)
    K = int(rel_freq * len(sorted_dims))
    n_dims = sum([d in np.argsort(-W_sigma_mean)[:K] for d in sorted_dims[:K]])
    if K:
        return float(n_dims / K)
    else:
        return 0


def compare_dimensions(Ws_mu: list, thresh: float, Ws_sigma=None) -> Tuple[np.ndarray]:
    N = max(Ws_mu[0].shape)
    rnd_perm = np.random.permutation(N)
    train_indices = rnd_perm[:int(N * .8)]
    test_indices = rnd_perm[int(N * .8):]
    loc_robustness_scores = []
    scale_robustness_scores = []
    for i, W_mu_i in enumerate(Ws_mu):
        for j, W_mu_j in enumerate(Ws_mu):
            if i != j:
                assert max(W_mu_i.shape) == max(
                    W_mu_j.shape), '\nNumber of items in weight matrices must align.\n'
                corrs = np.zeros(min(W_mu_i.shape))
                W_mu_i_train, W_mu_j_train = W_mu_i[:,
                                                    train_indices], W_mu_j[:, train_indices]
                W_mu_i_test, W_mu_j_test = W_mu_i[:,
                                                  test_indices], W_mu_j[:, test_indices]
                for k, w_i in enumerate(W_mu_i_train):
                    argmax = np.argmax([utils.pearsonr(w_i, w_j)
                                       for w_j in W_mu_j_train])
                    corrs[k] = utils.pearsonr(
                        W_mu_i_test[k], W_mu_j_test[argmax])
                sorted_dims = np.argsort(-corrs)
                rel_freq = robustness(corrs, thresh)
                loc_robustness_scores.append(rel_freq)
                if Ws_sigma:
                    W_sigma_i_test = Ws_sigma[i][test_indices]
                    scale_robustness_scores.append(uncertainty_estimates(
                        W_sigma_i_test, sorted_dims, rel_freq))
    avg_loc_robustness = np.mean(loc_robustness_scores)
    std_loc_robustness = np.std(loc_robustness_scores)
    avg_scale_robustness = np.mean(scale_robustness_scores)
    return avg_loc_robustness, std_loc_robustness, avg_scale_robustness


def estimate_redundancy_(Ws_mu: list) -> Tuple[float, float]:
    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        return (u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v))

    def get_redundant_pairs(W: np.ndarray, thresh: float = .9) -> int:
        w_combs = list(itertools.combinations(W, 2))
        cosine_sims = np.array([cosine(w_i, w_j)
                               for (w_i, w_j) in w_combs])
        n_redundant_pairs = np.where(cosine_sims > thresh, 1, 0).sum()
        return n_redundant_pairs

    def get_redundant_dims(W: np.ndarray, thresh: float = .9) -> int:
        n_redundant_dims = 0
        for i, w_i in enumerate(W):
            for j, w_j in enumerate(W):
                if i != j:
                    cos_sim = cosine(w_i, w_j)
                    if cos_sim > thresh:
                        n_redundant_dims += 1
                        print(
                            f'\nFound redundant dimension with cross-cosine similarity: {cos_sim.round(3)}.\n')
                        break
        return n_redundant_dims
    avg_redundant_pairs = np.mean(list(map(get_redundant_pairs, Ws_mu)))
    avg_redundant_dims = np.mean(list(map(get_redundant_dims, Ws_mu)))
    return avg_redundant_pairs, avg_redundant_dims


def compute_robustness(Ws_mu: list, Ws_sigma: list = None, thresh: float = .9):
    avg_loc_robustness, std_loc_robustness, avg_scale_robustness = compare_dimensions(
        Ws_mu, thresh, Ws_sigma)
    scores = {}
    scores['avg_loc_robustness'] = avg_loc_robustness
    scores['std_loc_robustness'] = std_loc_robustness
    scores['avg_scale_robustness'] = avg_scale_robustness
    scores['avg_ndims'] = avg_ndims(Ws_mu)
    scores['hist'] = list(map(lambda W: W.shape[0], Ws_mu))
    scores['std_ndims'] = std_ndims(Ws_mu)
    n_redundant_pairs, n_redundant_dims = estimate_redundancy_(Ws_mu)
    scores['n_redundant_pairs'] = n_redundant_pairs
    scores['n_redundant_dims'] = n_redundant_dims
    return scores


def get_model_paths(PATH: str):
    paths = []
    for root, _, files in os.walk(PATH, followlinks=True):
        files = sorted(list(filter(lambda f: re.search(r'tar$', f), files)))
        if files:
            for f in files:
                if f == files[-1]:
                    paths.append('/'.join(root.split('/')[:-1]))
    return paths


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


def compute_pvals(W_mu: np.ndarray, W_sigma: np.ndarray) -> np.ndarray:
    return np.array([norm.cdf(0, W_mu[:, j], W_sigma[:, j]) for j in range(W_mu.shape[1])])


def fdr_correction(p_vals: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.array(list(map(lambda p: multipletests(p, alpha=alpha, method='fdr_bh')[0], p_vals)))


def get_importance(rejections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(list(map(sum, rejections)))[:, np.newaxis]


def get_cluster_combinations(n_clusters: int, c_min: int = 1):
    combinations = []
    for k in range(c_min, n_clusters):
        combinations.extend(list(itertools.combinations(range(n_clusters), k)))
    return combinations


def get_cluster_indices(clusters: np.ndarray, subset: Tuple[int]) -> np.ndarray:
    return np.hstack([np.where(clusters == k)[0] for k in subset])


def get_best_subset_and_noise(val_losses: np.ndarray, clusters: np.ndarray, cluster_combs: List[Tuple[int]], n_clusters: int,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    best_subset = cluster_combs[np.argmin(val_losses)]
    remaining_subset = tuple(i for i in range(
        n_clusters) if i not in best_subset)
    best = get_cluster_indices(clusters, best_subset)
    noise = get_cluster_indices(clusters, remaining_subset)
    return best, noise


def pruning(
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
        best_subset, _ = get_best_subset_and_noise(
            val_losses, clusters, cluster_combs, n_clusters)
        W_loc = W_loc[:, best_subset]
        W_scale = W_scale[:, best_subset]
        best_model = pruned_models[np.argmin(val_losses)]
        pruning_loss = np.min(val_losses)
    else:
        best_model = model
        pruning_loss, _ = trainer.evaluate(pruning_batches)
    return W_loc.T, W_scale, best_model, pruning_loss


def evaluate_models(
    results_dir: str,
    modality: str,
    task: str,
    n_items: int,
    latent_dim: int,
    optim: str,
    prior: str,
    spike: float,
    slab: float,
    pi: float,
    thresh: float,
    device: torch.device,
    batch_size: int,
    triplets_dir: str,
    n_components: int,
    mc_samples: int,
    things: bool = False,
) -> None:
    in_path = os.path.join(results_dir, modality,
                           f'{latent_dim}d', optim, prior, str(spike), str(slab), str(pi))
    model_paths = get_model_paths(in_path)
    Ws_loc_best, Ws_scale_best = [], []
    _, pruning_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir, val_set='pruning_set', inference=False)
    _, tuning_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir, val_set='tuning_set', inference=False)
    pruning_batches = utils.load_batches(
        train_triplets=None, test_triplets=pruning_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    tuning_batches = utils.load_batches(
        train_triplets=None, test_triplets=tuning_triplets, n_items=n_items, batch_size=batch_size, inference=True)
    pruning_losses = np.zeros(len(model_paths))
    tuning_losses = np.zeros(len(model_paths))
    for i, model_path in enumerate(model_paths):
        print(f'Currently pruning model: {i+1}\n')
        try:
            model = VICE(prior=prior, in_size=n_items,
                         out_size=latent_dim, init_weights=True)
            model = utils.load_model(
                model=model, PATH=model_path, device=device)
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
                spike=spike,
                slab=slab,
                pi=pi,
                steps=None,
                model_dir=os.path.join(model_path, 'model'),
                results_dir=results_dir,
                device=device)
            W_loc_best, W_scale_best, pruned_model, pruning_loss = pruning(
                trainer=trainer, pruning_batches=pruning_batches, n_components=n_components,  device=device, things=things)
            trainer.model = pruned_model
            tuning_loss, _ = trainer.evaluate(tuning_batches)
            pruning_losses[i] += pruning_loss
            tuning_losses[i] += tuning_loss
        except FileNotFoundError:
            raise Exception(f'Could not find final weights for {model_path}\n')
        Ws_loc_best.append(W_loc_best)
        Ws_scale_best.append(W_scale_best)

    model_robustness_best_subset = compute_robustness(
        Ws_mu=Ws_loc_best, Ws_sigma=Ws_scale_best, thresh=thresh)
    print(
        f"\nRobustness scores for latent dim = {latent_dim}: {model_robustness_best_subset}\n")

    out_path = pjoin(in_path, 'robustness_scores', str(thresh))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(pjoin(out_path, 'robustness.txt'), 'wb') as f:
        f.write(pickle.dumps(model_robustness_best_subset))

    with open(os.path.join(in_path, 'tuning_cross_entropies.npy'), 'wb') as f:
        np.save(f, tuning_losses)

    with open(os.path.join(in_path, 'pruning_cross_entropies.npy'), 'wb') as f:
        np.save(f, pruning_losses)


if __name__ == '__main__':
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

    evaluate_models(
        results_dir=args.results_dir,
        modality=args.modality,
        task=args.task,
        n_items=args.n_items,
        latent_dim=args.latent_dim,
        optim=args.optim,
        prior=args.prior,
        spike=args.spike,
        slab=args.slab,
        pi=args.pi,
        thresh=args.thresh,
        device=args.device,
        batch_size=args.batch_size,
        triplets_dir=args.triplets_dir,
        n_components=args.n_components,
        mc_samples=args.mc_samples,
        things=args.things,
    )
