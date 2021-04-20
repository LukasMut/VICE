#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
import random
import re
import torch
import utils
import itertools
import shutil
import copy

import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from models.model import VSPoSE
from os.path import join as pjoin
from typing import Tuple, List, Dict, Any, Iterator
from sklearn import mixture
from scipy.stats import laplace
from statsmodels.stats.multitest import multipletests

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--modality', type=str,
        help='current modality (e.g., behavioral, synthetic)')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--version', type=str,
        choices=['deterministic', 'variational'],
        help='deterministic or variational version of SPoSE')
    aa('--dim', type=int, default=100,
        help='latent dimensionality of VSPoSE embedding matrices')
    aa('--thresh', type=float, default=0.85,
        choices=[0.75, 0.8, 0.85, 0.9, 0.95],
        help='examine fraction of dimensions across models that is above threshold (corresponds to Pearson correlation)')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--triplets_dir', type=str, default=None,
        help='directory from where to load triplets data')
    aa('--n_components', type=int, nargs='+', default=None,
        help='number of clusters/modes in Gaussian mixture model')
    aa('--n_samples', type=int, default=None,
        choices=[5, 10, 15, 20, 25],
        help='number of samples to use for MCMC sampling during validation')
    aa('--device', type=str,
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def avg_ndims(Ws_mu:list) -> np.ndarray:
    return np.ceil(np.mean(list(map(lambda w: min(w.shape), Ws_mu))))

def std_ndims(Ws_mu:list) -> np.ndarray:
    return np.std(list(map(lambda w: min(w.shape), Ws_mu)))

def robustness(corrs:np.ndarray, thresh:float) -> float:
    return len(corrs[corrs>thresh])/len(corrs)

def uncertainty_estimates(W_b:np.ndarray, sorted_dims:np.ndarray, rel_freq:float) -> float:
    W_b_mean = np.mean(W_b, axis=0)
    assert len(W_b_mean) == min(W_b.shape)
    K = int(rel_freq*len(sorted_dims))
    n_dims = sum([d in np.argsort(-W_b_mean)[:K] for d in sorted_dims[:K]])
    return float(n_dims / K)

"""
def nmf_grid_search(Ws_mu:list, n_components:np.ndarray, k_folds:int=2, n_repeats:int=5, rnd_seed:int=42) -> Tuple[np.ndarray]:
    np.random.seed(rnd_seed)
    rkf = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=rnd_seed)
    W_held_out = Ws_mu.pop(np.random.choice(len(Ws_mu))).T
    X = np.concatenate(Ws_mu, axis=0).T
    X = X[:, np.random.permutation(X.shape[1])]
    avg_r2_scores = np.zeros(len(n_components))
    W_nmfs = []
    for j, n_comp in enumerate(n_components):
        nmf = NMF(n_components=n_comp, init='nndsvd', max_iter=5000, random_state=rnd_seed)
        W_nmf = nmf.fit_transform(X)
        nnls_reg = LinearRegression(positive=True)
        r2_scores = np.zeros(int(k_folds * n_repeats))
        for k, (train_idx, test_idx) in enumerate(rkf.split(W_nmf)):
            X_train, X_test = W_nmf[train_idx], W_nmf[test_idx]
            y_train, y_test = W_held_out[train_idx], W_held_out[test_idx]
            nnls_reg.fit(X_train, y_train)
            y_pred = nnls_reg.predict(X_test)
            r2_scores[k] = r2_score(y_test, y_pred)
        avg_r2_scores[j] = np.mean(r2_scores)
        W_nmfs.append(W_nmf.T)
    W_nmf_final = W_nmfs[np.argmax(avg_r2_scores)]
    W_nmf_final = W_nmf_final[np.argsort(-np.linalg.norm(W_nmf_final, ord=1, axis=1))]
    return W_nmf_final, avg_r2_scores
"""

def compare_dimensions(Ws_mu:list, thresh:float, Ws_b=None) -> Tuple[np.ndarray]:
    N = max(Ws_mu[0].shape)
    rnd_perm = np.random.permutation(N)
    train_indices = rnd_perm[:int(N*.8)]
    test_indices = rnd_perm[int(N*.8):]
    loc_robustness_scores = []
    scale_robustness_scores = []
    for i, W_mu_i in enumerate(Ws_mu):
        for j, W_mu_j in enumerate(Ws_mu):
            if i != j:
                assert max(W_mu_i.shape) == max(W_mu_j.shape), '\nNumber of items in weight matrices must align.\n'
                corrs = np.zeros(min(W_mu_i.shape))
                W_mu_i_train, W_mu_j_train = W_mu_i[:, train_indices], W_mu_j[:, train_indices]
                W_mu_i_test, W_mu_j_test = W_mu_i[:, test_indices], W_mu_j[:, test_indices]
                for k, w_i in enumerate(W_mu_i_train):
                    argmax = np.argmax([utils.pearsonr(w_i, w_j) for w_j in W_mu_j_train])
                    corrs[k] = utils.pearsonr(W_mu_i_test[k], W_mu_j_test[argmax])
                sorted_dims = np.argsort(-corrs)
                rel_freq = robustness(corrs, thresh)
                loc_robustness_scores.append(rel_freq)
                if Ws_b:
                    W_b_i_test = Ws_b[i][test_indices]
                    scale_robustness_scores.append(uncertainty_estimates(W_b_i_test, sorted_dims, rel_freq))
    max_loc_robustness = np.max(loc_robustness_scores)
    avg_loc_robustness = np.mean(loc_robustness_scores)
    avg_scale_robustness = np.mean(scale_robustness_scores)
    return max_loc_robustness, avg_loc_robustness, avg_scale_robustness

def estimate_redundancy_(Ws_mu:list) -> Tuple[float, float]:
    def cosine_sim(x:np.ndarray, y:np.ndarray) -> float:
        return (x @ y) / (np.linalg.norm(x)*np.linalg.norm(y))
    def get_redundant_pairs(W:np.ndarray, thresh:float=.9) -> int:
        w_combs = list(itertools.combinations(W, 2))
        cosine_sims = np.array([cosine_sim(w_i, w_j) for (w_i, w_j) in w_combs])
        n_redundant_pairs = np.where(cosine_sims > thresh, 1, 0).sum()
        return n_redundant_pairs
    def get_redundant_dims(W:np.ndarray, thresh:float=.9) -> int:
        n_redundant_dims = 0
        for i, w_i in enumerate(W):
            for j, w_j in enumerate(W):
                if i != j:
                    cos_sim = cosine_sim(w_i, w_j)
                    if cos_sim > thresh:
                        n_redundant_dims += 1
                        print(f'\nFound redundant dimension with cross-cosine similarity: {cos_sim.round(3)}.\n')
                        break
        return n_redundant_dims
    avg_redundant_pairs = np.mean(list(map(get_redundant_pairs, Ws_mu)))
    avg_redundant_dims = np.mean(list(map(get_redundant_dims, Ws_mu)))
    return avg_redundant_pairs, avg_redundant_dims

def compute_robustness(Ws_mu:list, Ws_b:list=None, thresh:float=.9):
    max_loc_robustness, avg_loc_robustness, avg_scale_robustness = compare_dimensions(Ws_mu, thresh, Ws_b)
    scores = {}
    scores['max_loc_robustness'] = max_loc_robustness
    scores['avg_loc_robustness'] = avg_loc_robustness
    scores['avg_scale_robustness'] = avg_scale_robustness
    scores['avg_sparsity'] = utils.avg_sparsity(Ws_mu)
    scores['avg_ndims'] = avg_ndims(Ws_mu)
    scores['std_ndims'] = std_ndims(Ws_mu)
    n_redundant_pairs, n_redundant_dims = estimate_redundancy_(Ws_mu)
    scores['n_redundant_pairs'] = n_redundant_pairs
    scores['n_redundant_dims'] = n_redundant_dims
    return scores

def del_paths_(paths:List[str]) -> None:
    for path in paths:
        idx = 2 if re.search(r'SPoSE', path) else 1
        shutil.rmtree(path)
        plots_path = path.split('/')
        plots_path[idx] = 'plots'
        plots_path = '/'.join(plots_path)
        shutil.rmtree(plots_path)

def get_best_hypers_(PATH:str) -> Tuple[str, float]:
    paths, results = [], []
    for root, dirs, files in os.walk(PATH):
        for name in files:
            if name.endswith('.json'):
                paths.append(root)
                with open(os.path.join(root, name), 'r') as f:
                    results.append(json.load(f)['val_loss'])
    argmin_loss = np.argmin(results)
    best_model = paths.pop(argmin_loss)
    print(f'Best params: {best_model}\n')
    del_paths_(paths)
    return best_model

def get_model_paths_(PATH:str) -> List[str]:
    #model_paths = [get_best_hypers_(os.path.join(PATH, d.name)) for d in os.scandir(PATH) if d.is_dir() and d.name[-2:].isdigit()]
    model_paths = []
    for d in os.scandir(PATH):
        if d.is_dir() and d.name[-2:].isdigit():
            try:
                best_model = get_best_hypers_(os.path.join(PATH, d.name))
                model_paths.append(best_model)
            except ValueError:
                print(f'Could not find results for {d.name}\n')
                pass
    return model_paths

def prune_weights(model:Any, indices:torch.Tensor) -> Any:
    for m in model.parameters():
        m.data = m.data[indices]
    return model

def fit_gmm(X:np.ndarray, n_components:List[int]) -> Tuple[int, Any]:
    gmms, scores = [], []
    cv_types = ['spherical', 'diag', 'full']
    hyper_combs = list(itertools.product(cv_types, n_components))
    for comb in hyper_combs:
        gmm = mixture.GaussianMixture(n_components=comb[1], covariance_type=comb[0])
        gmm.fit(X)
        gmms.append(gmm)
        scores.append(gmm.bic(X))
    clusters = gmms[np.argmin(scores)].predict(X)
    n_clusters = hyper_combs[np.argmin(scores)][1]
    return clusters, n_clusters

def compute_pvals(W_mu:np.ndarray, W_b:np.ndarray) -> np.ndarray:
    return np.array([laplace.cdf(0, W_mu[:, j], W_b[:, j]) for j in range(W_mu.shape[1])])

def fdr_correction(p_vals:np.ndarray, alpha:float=0.01) -> np.ndarray:
    return np.array(list(map(lambda p: multipletests(p, alpha=alpha, method='fdr_bh')[0], p_vals)))

def get_importance(rejections:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(list(map(sum, rejections)))[:, np.newaxis]

def pruning(
            model:Any,
            task:str,
            val_batches:Iterator[torch.Tensor],
            n_components:List[int],
            n_samples:int,
            device:torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W_mu, W_b = utils.load_weights(model)
    W_mu = W_mu[sortindex]
    W_b = W_b[sortindex]
    importance = get_importance(fdr_correction(compute_pvals(W_mu, W_b)))
    clusters, n_clusters = fit_gmm(importance, n_components)
    val_accs = np.zeros(n_clusters)
    for k in range(n_clusters):
        indices = torch.from_numpy(np.where(clusters==k)[0]).type(torch.LongTensor).to(device)
        model_copy = copy.deepcopy(model)
        model_pruned = prune_weights(model_copy, indices)
        _, val_acc = utils.validation(
                                      model=model_pruned,
                                      task=task,
                                      val_batches=val_batches,
                                      version='variational',
                                      device=device,
                                      n_samples=n_samples,
                                )
        val_accs[k] += val_acc
    W_mu = W_mu[:, np.where(clusters!=np.argmin(val_accs))[0]].cpu().numpy()
    W_b = W_b[:, np.where(clusters!=np.argmin(val_accs))[0]].cpu().numpy()
    return W_mu.T, W_b

def evaluate_models(
                    results_dir:str,
                    modality:str,
                    version:str,
                    dim:int,
                    thresh:float,
                    device:torch.device,
                    task=None,
                    batch_size=None,
                    triplets_dir=None,
                    n_components=None,
                    n_samples=None,
                    ) -> None:
    N_ITEMS = 1854
    PATH = os.path.join(results_dir, modality, version, f'{dim}d')
    model_paths = get_model_paths_(PATH)
    Ws_mu, Ws_b = [], []
    for model_path in model_paths:
        if version == 'variational':
            try:
                model = VSPoSE(N_ITEMS, dim)
                lmbda = float(model_path.split('/')[-2])
                model = utils.load_model(model, model_path, device)
                _, test_triplets = utils.load_data(device=device, triplets_dir=triplets_dir, inference=False)
                val_batches = utils.load_batches(train_triplets=None, test_triplets=test_triplets, n_items=N_ITEMS, batch_size=batch_size, inference=True)
                W_mu, W_b = pruning(model=model, task=task, val_batches=val_batches, n_components=n_components, n_samples=n_samples, device=device)
                #W_mu, W_b = utils.load_weights(model)
                #W_mu, W_b = W_mu.numpy(), W_b.numpy()
                #W_mu, W_b = W_mu[sortindex], W_b[sortindex]
                #sorted_dims, klds_sorted = utils.compute_kld(model, lmbda, aggregate=True, reduction='max')
                #W_mu, W_b = W_mu[:, sorted_dims], W_b[:, sorted_dims]
                #W_mu = W_mu[:, :utils.kld_cut_off(np.log(klds_sorted))].T
            except FileNotFoundError:
                raise Exception(f'Could not find final weights for {model_path}\n')
            Ws_b.append(W_b)
        else:
            try:
                W_mu = utils.load_final_weights(model_path, version)
                W_mu = W_mu[sortindex]
                W_mu = utils.remove_zeros(W_mu.T)
            except FileNotFoundError:
                raise Exception(f'Could not find final weights for {model_path}\n')
        Ws_mu.append(W_mu)

    model_robustness = compute_robustness(Ws_mu, Ws_b=Ws_b, thresh=thresh)
    print(f"\nRobustness scores for latent dim = {dim}: {model_robustness}\n")

    out_path = pjoin(PATH, 'robustness_scores', str(thresh))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(pjoin(out_path, 'robustness.txt'), 'wb') as f:
        f.write(pickle.dumps(model_robustness))

if __name__ == '__main__':
    args = parseargs()
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    _, sortindex = utils.load_inds_and_item_names()
    evaluate_models(
                    results_dir=args.results_dir,
                    modality=args.modality,
                    task=args.task,
                    version=args.version,
                    dim=args.dim,
                    thresh=args.thresh,
                    device=args.device,
                    batch_size=args.batch_size,
                    triplets_dir=args.triplets_dir,
                    n_components=args.n_components,
                    n_samples=args.n_samples,
                    )
