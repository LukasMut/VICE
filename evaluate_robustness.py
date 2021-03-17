#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
import pickle
import random
import re
import torch
import utils
import itertools

import numpy as np

from collections import defaultdict
from models.model import VSPoSE, SPoSE
from os.path import join as pjoin
from typing import Tuple, List, Dict
from plotting import plot_val_accs_across_seeds
from sklearn.model_selection import RepeatedKFold

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

def aggregate_val_accs(PATH:str, rnd_seeds:List[int]) -> np.ndarray:
    val_accs = np.zeros(len(rnd_seeds))
    for k, rnd_seed in enumerate(rnd_seeds):
        with open(pjoin(PATH, f'seed{rnd_seed:02d}', 'results.json'), 'rb') as results:
            val_accs[k] = json.load(results)[''.join(('seed', '_', str(rnd_seed)))]['val_acc']
    return np.mean(val_accs)

def evaluate_models(modality:str, version:str, lmbda:float, thresh:float, device:torch.device) -> None:
    _, sortindex = utils.load_inds_and_item_names()
    #nmf_components = np.arange(50, 200, 10)
    dims = np.array([100, 200])
    #lmbdas = np.arange(5.75e+3, 7.5e+3, 2.5e+2) if version == 'variational' else np.array([0.008, 0.0085, 0.0088])
    lmbdas = np.arange(6.5e+3, 7.5e+3, 2.5e+2) if version == 'variational' else np.array([0.008, 0.0085, 0.0088])
    results_dir = './results'
    N_ITEMS = 1854
    for dim in dims:
        avg_val_accs = np.zeros(len(lmbdas))
        for l, lmbda in enumerate(lmbdas):
            if version == 'variational':
                if dim == 200:
                    lmbda *= 2
            PATH = pjoin(results_dir, modality, version, 'human', f'{dim}d', str(lmbda))
            rnd_seeds = [dir for dir in os.listdir(PATH) if dir[-2:].isdigit()]
            rnd_seeds = sorted(list(map(lambda str:int(str.replace('seed', '')), rnd_seeds)))
            avg_val_accs[l] = aggregate_val_accs(PATH, rnd_seeds)
            Ws_mu, Ws_b = [], []
            for rnd_seed in rnd_seeds:
                if version == 'variational':
                    #initialise a variational version of SPoSE
                    model = VSPoSE(in_size=N_ITEMS, out_size=dim, init_weights=True, init_method='normal', device=device, rnd_seed=rnd_seed)
                else:
                    model = SPoSE(in_size=N_ITEMS, out_size=dim)

                try:
                    #load weights of converged VSPoSE model with dSPoSE initialisation
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

                if version == 'variational':
                #sort VSPoSE dimensions in descending order according to their KLDs and find the top_k dims
                    sorted_dims, _ = utils.compute_kld(model, lmbda=lmbda, aggregate=True, reduction='max')
                    W_mu, W_b = utils.load_weights(model, 'variational')
                    W_mu, W_b = W_mu.numpy(), W_b.numpy()
                    W_mu, W_b = W_mu[sortindex], W_b[sortindex]
                    W_mu = W_mu[:, sorted_dims]
                    W_b = W_b[:, sorted_dims]
                    Ws_b.append(W_b)
                else:
                    _, W_mu = utils.sort_weights(model, aggregate=False)
                    W_mu = W_mu[sortindex]

                W_mu = utils.remove_zeros(W_mu.T)
                Ws_mu.append(W_mu)

            model_robustness = compute_robustness(Ws_mu, Ws_b=Ws_b, thresh=thresh)

            print(f"\nRobustness scores for lambda = {lmbda:.5f} and latent dim = {dim}: {model_robustness}\n")

            #Ws_mu_nmf = Ws_mu.copy()
            #W_nmf_final, mean_r2_scores = nmf_grid_search(Ws_mu_nmf, n_components=nmf_components)

            #Ws_mu_split1 = Ws_mu.copy()[:len(Ws_mu)]
            #Ws_mu_split2 = Ws_mu.copy()[len(Ws_mu):]

            #W_nmf_split1, _ = nmf_grid_search(Ws_mu_split1, n_components=nmf_components)
            #W_nmf_split2, _ = nmf_grid_search(Ws_mu_split2, n_components=nmf_components)


            out_path = pjoin(PATH, 'robustness_scores', str(thresh))
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with open(pjoin(out_path, 'robustness.txt'), 'wb') as f:
                f.write(pickle.dumps(model_robustness))

        plots_dir = pjoin('./plots', modality, version, f'{dim}d')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plot_val_accs_across_seeds(plots_dir, lmbdas, avg_val_accs)

if __name__ == '__main__':
    #seed random number generator
    random.seed(42)
    np.random.seed(42)
    #parse os argument variables
    modality = sys.argv[1]
    version = sys.argv[2]
    lmbda = float(sys.argv[3])
    thresh = float(sys.argv[4])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate_models(modality=modality, version=version, lmbda=lmbda, thresh=thresh, device=device)
