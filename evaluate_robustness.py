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
import model

import numpy as np

from typing import Tuple, List

os.environ['PYTHONIOENCODING'] = 'UTF-8'
# number of cores used per Python process (set to 2 if HT is enabled, else keep 1)
os.environ['OMP_NUM_THREADS'] = '1'


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--results_dir', type=str,
        help='results directory (root directory for models)')
    aa('--n_objects', type=int,
        help='number of unique objects/items/stimuli in dataset')
    aa('--init_dim', type=int, default=100,
        help='initial latent dimensionality of VICE embedding(s)')
    aa('--thresh', type=float, default=0.8,
        choices=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help='reproducibility threshold (0.8 used in the ICLR paper)')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets in each mini-batch')
    aa('--optim', type=str, metavar='o', default='adam',
        choices=['adam', 'adamw', 'sgd'],
        help='optimizer that was used to train VICE')
    aa('--prior', type=str, metavar='p', default='gaussian',
        choices=['gaussian', 'laplace'],
        help='whether to use a Gaussian or Laplacian mixture for the spike-and-slab prior')
    aa('--spike', type=float,
        help='sigma of spike distribution')
    aa('--slab', type=float,
        help='sigma of slab distribution')
    aa('--pi', type=float,
        help='probability value with which to sample from the spike')
    aa('--triplets_dir', type=str,
        help='path/to/triplets/data')
    aa('--mc_samples', type=int, default=5,
        choices=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of weight samples used in Monte Carlo (MC) sampling')
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


def get_model_paths(PATH: str) -> List[str]:
    regex = r'(?=^model)(?=.*epoch)(?=.*tar$)'
    paths = []
    for root, _, files in os.walk(PATH, followlinks=True):
        files = sorted(
            list(filter(lambda f: re.compile(regex).search(f), files)))
        if files:
            paths.append('/'.join(root.split('/')[:-1]))
    return paths


def prune_weights(model: model.VICE, indices: np.ndarray) -> model.VICE:
    for p in model.parameters():
        p.data = p.data[torch.from_numpy(indices)]
    return model


def pruning(model: model.VICE, alpha: float = .05, k: int = 5,
            ) -> Tuple[torch.Tensor, torch.Tensor, model.VICE]:
    params = model.detached_params
    loc = params['loc']
    scale = params['scale']
    p_vals = utils.compute_pvals(loc, scale)
    rejections = utils.fdr_corrections(p_vals, alpha)
    importance = utils.get_importance(rejections).ravel()
    signal = np.where(importance > k)[0]
    pruned_loc = loc[:, signal]
    pruned_scale = scale[:, signal]
    pruned_model = prune_weights(model, signal)
    return loc, pruned_loc.T, pruned_scale, pruned_model


def evaluate_models(
    results_dir: str,
    n_objects: int,
    init_dim: int,
    optim: str,
    prior: str,
    spike: float,
    slab: float,
    pi: float,
    thresh: float,
    device: torch.device,
    batch_size: int,
    triplets_dir: str,
    mc_samples: int,
    k: int = 5,
) -> None:
    in_path = os.path.join(results_dir,
                           f'{init_dim}d', optim, prior, str(spike), str(slab), str(pi))
    model_paths = get_model_paths(in_path)
    _, val_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir, inference=False)
    val_batches = utils.load_batches(
        train_triplets=None, test_triplets=val_triplets, n_objects=n_objects, batch_size=batch_size, inference=True)
    locs, pruned_locs, pruned_scales = [], [], []
    val_losses = np.zeros(len(model_paths), dtype=np.float32)
    for i, model_path in enumerate(model_paths):
        print(f'Currently pruning and evaluating model: {i+1}\n')
        try:
            vice = model.VICE(
                k=k,
                n_train=None,
                burnin=None,
                ws=None,
                n_objects=n_objects,
                init_dim=init_dim,
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
                device=device,
                init_weights=True)
            vice = utils.load_model(
                model=vice, PATH=model_path, device=device)
        except FileNotFoundError:
            raise Exception(f'Could not find params for {model_path}\n')
        loc, pruned_loc, pruned_scale, pruned_vice = pruning(vice)
        val_loss, _ = pruned_vice.evaluate(val_batches)
        val_losses[i] += val_loss
        locs.append(loc)
        pruned_locs.append(pruned_loc)
        pruned_scales.append(pruned_scale)

    model_robustness_best_subset = compute_robustness(
        Ws_mu=pruned_locs, Ws_sigma=pruned_scales, thresh=thresh)
    print(
        f"\nRobustness scores: {model_robustness_best_subset}\n")

    out_path = os.path.join(in_path, 'robustness_scores', str(thresh))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, 'robustness.txt'), 'wb') as f:
        f.write(pickle.dumps(model_robustness_best_subset))

    with open(os.path.join(in_path, 'val_entropies.npy'), 'wb') as f:
        np.save(f, val_losses)

    print('\nSaving mean embedding with the lowest cross-entropy error on the validation set.\n')
    final_embedding_unpruned = locs[np.argmin(val_losses)]
    final_embedding_pruned = pruned_locs[np.argmin(val_losses)].T
    final_embedding_pruned[
            :, np.argsort(-np.linalg.norm(final_embedding_pruned, axis=0, ord=1))
        ]
    with open(os.path.join(in_path, 'final_embedding_pruned.npy'), 'wb') as f:
        np.save(f, final_embedding_pruned)
    
    with open(os.path.join(in_path, 'final_embedding_unpruned.npy'), 'wb') as f:
        np.save(f, final_embedding_unpruned)


if __name__ == '__main__':
    args = parseargs()
    random.seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    evaluate_models(
        results_dir=args.results_dir,
        n_objects=args.n_objects,
        init_dim=args.init_dim,
        optim=args.optim,
        prior=args.prior,
        spike=args.spike,
        slab=args.slab,
        pi=args.pi,
        thresh=args.thresh,
        device=args.device,
        batch_size=args.batch_size,
        triplets_dir=args.triplets_dir,
        mc_samples=args.mc_samples,
    )
