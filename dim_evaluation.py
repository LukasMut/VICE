#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import torch

import numpy as np

from models.model import VICE, SPoSE
from plotting import plot_aggregated_weights, plot_weight_violins, plot_pruning_results
from utils import  BatchGenerator, compute_kld, get_cut_off, load_data, load_model, prune_weights, sort_weights, validation

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--version', type=str, default='variational',
        choices=['deterministic', 'variational'],
        help='whether to apply a deterministic or variational version of SPoSE')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--modality', type=str, default='behavioral',
        help='define for which modality task should be performed')
    aa('--init', type=str, default=None,
        choices=[None, 'dSPoSE', 'normal'],
        help='initialisation of variational SPoSE model')
    aa('--data', type=str, default='human',
        choices=['dspose_init', 'human', 'synthetic', ''],
        help='define whether to use synthetically created triplet choices or human choices')
    aa('--reduction', type=str, default='sum',
        choices=['sum', 'max', 'l1_norm'],
        help='function applied to aggregate KL divergences across dimensions')
    aa('--results_dir', type=str, default='./results',
        help='optional specification of results directory (if not provided will resort to ./results/)')
    aa('--plot_dir', type=str, default='./plots',
        help='optional specification of directory for plots (if not provided will resort to ./plots/)')
    aa('--embed_dim', metavar='D', type=int, default=100,
        choices=[100, 200, 300, 400],
        help='dimensionality of the embedding matrix')
    aa('--batch_size', metavar='B', type=int, default=128,
        choices=[16, 32, 50, 64, 100, 128, 256],
        help='number of triplets in each mini-batch')
    aa('--n_items', type=int, default=1854,
        help='number of unique items in dataset')
    aa('--n_samples', type=int, default=None,
        help='specify over how many samples output logits should be averaged in variational SPoSE (at inference time)')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def run(
        task:str,
        results_dir:str,
        plot_dir:str,
        version:str,
        modality:str,
        init:str,
        data:str,
        reduction:str,
        embed_dim:int,
        n_items:int,
        batch_size:int,
        device:torch.device,
        rnd_seed:int,
        n_samples=None,
        ):
    #load test triplets
    _, test_triplets = load_data(device=device, triplets_dir=os.path.join('triplets', modality))
    #initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    #get mini-batches for validation
    val_batches = BatchGenerator(I=I, dataset=test_triplets, batch_size=batch_size, sampling_method=None, p=None)
    #fraction of encoder weights and biases to be pruned before inference (to reduce latent dimensionality)
    pruning_fracs = np.arange(0., 0.95, 0.05)

    if version == 'variational':
        #lambdas denote the strength of the prior (1/lambda = b = scale of the prior distribution; mu = mode of the prior distributio)
        if embed_dim == 100:
            lambdas = np.arange(5.5e+3, 8.5e+3, 1e+3)
        elif embed_dim == 200:
            lambdas = np.arange(1.3e+4, 1.7e+4, 1e+3)
        else:
            pass
    else:
        lambdas = np.array([0.008])

    for lmbda in lambdas:
        #initialise model
        if version == 'variational':
            assert isinstance(init, str), 'define initialisation of VICE'
            model = VICE(in_size=n_items, out_size=embed_dim, init_weights=True, init_method=init, device=device, rnd_seed=rnd_seed)
        else:
            model = SPoSE(in_size=n_items, out_size=embed_dim)
        #load weights of pretrained model
        model = load_model(
                           model=model,
                           results_dir=results_dir,
                           modality=modality,
                           version=version,
                           data=data,
                           dim=embed_dim,
                           lmbda=lmbda,
                           rnd_seed=rnd_seed,
                           device=device,
                           )
        if version == 'variational':
            #sort columns of Ws (i.e., dimensions) in VICE according to their KL divergences in descending order
            sorted_dims, klds_sorted = compute_kld(model, lmbda, aggregate=True, reduction=reduction)
        else:
            #sort columns of W (i.e., dimensions) in SPoSE according to their l1-norms in descending
            sorted_dims, l1_sorted = sort_weights(model, aggregate=True)

        #elbow plot of KL divergences aggregated over items
        plot_aggregated_weights(
                                W_aggregated=klds_sorted if version == 'variational' else l1_sorted,
                                plot_dir=plot_dir,
                                rnd_seed=rnd_seed,
                                modality=modality,
                                version=version,
                                data=data,
                                dim=embed_dim,
                                lmbda=lmbda,
                                reduction=reduction,
                                )
        if version == 'variational':
            #sort columns of Ws (i.e., dimensions) in VICE according to their KL divergences in descending order
            _, klds_sorted = compute_kld(model, lmbda, aggregate=False)
        else:
            #sort columns of W (i.e., dimensions) in SPoSE according to their l1-norms in descending
            _, W_sorted = sort_weights(model, aggregate=False)

        #violinplot of KL divergences across all items and latent dimensions
        plot_weight_violins(
                            W_sorted=klds_sorted if version == 'variational' else W_sorted,
                            plot_dir=plot_dir,
                            rnd_seed=rnd_seed,
                            modality=modality,
                            version=version,
                            data=data,
                            dim=embed_dim,
                            lmbda=lmbda,
                            reduction=reduction,
                            )

        #examine model performance as a function of pruned weights fraction
        results = []

        for prune_frac in pruning_fracs:
            #initialise model
            if version == 'variational':
                model = VICE(in_size=n_items, out_size=embed_dim)
            else:
                model = SPoSE(in_size=n_items, out_size=embed_dim)
            #load weights of pretrained model
            model = load_model(
                               model=model,
                               results_dir=results_dir,
                               modality=modality,
                               version=version,
                               data=data,
                               dim=embed_dim,
                               lmbda=lmbda,
                               rnd_seed=rnd_seed,
                               device=device,
                               )
            #move model to current device
            model.to(device)
            #prune weights
            model = prune_weights(
                                  model=model,
                                  version=version,
                                  indices=sorted_dims,
                                  fraction=(1-prune_frac),
                                  )
            _, val_acc = validation(
                                    model=model,
                                    val_batches=val_batches,
                                    version=version,
                                    task=task,
                                    device=device,
                                    embed_dim=embed_dim,
                                    batch_size=batch_size,
                                    n_samples=n_samples,
                                    )
            results.append((int(prune_frac * 100), float(val_acc * 100)))

        #plot validation performance as a function of pruned weights fraction
        plot_pruning_results(
                            results=results,
                            plot_dir=plot_dir,
                            rnd_seed=rnd_seed,
                            modality=modality,
                            version=version,
                            data=data,
                            dim=embed_dim,
                            lmbda=lmbda,
                            reduction=reduction,
                            )

if __name__ == '__main__':
    #parse all arguments and set random seeds
    args = parseargs()

    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    #set device
    device = torch.device(args.device)

    #some variables to debug / potentially resolve CUDA problems
    if device == torch.device('cuda:0'):
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)

    elif device == torch.device('cuda:1') or device == torch.device('cuda'):
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(1)

    print(f'PyTorch CUDA version: {torch.version.cuda}')
    print()

    run(
        task=args.task,
        results_dir=args.results_dir,
        plot_dir=args.plot_dir,
        version=args.version,
        modality=args.modality,
        data=args.data,
        init=args.init,
        reduction=args.reduction,
        embed_dim=args.embed_dim,
        n_items=args.n_items,
        batch_size=args.batch_size,
        device=args.device,
        rnd_seed=args.rnd_seed,
        n_samples=args.n_samples,
        )
