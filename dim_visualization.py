#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import torch
import utils

import numpy as np
import matplotlib.pyplot as plt

from models.model import VSPoSE, SPoSE
from plotting import plot_top_k_dim_correlations, plot_vspose_along_dpose_pcs, plot_vspose_against_dpose_pcs, plot_imgs_along_dimensions
from plotting import visualize_dims_across_modalities
from sklearn.decomposition import PCA
from typing import Tuple, List

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--modality', type=str, default='behavioral',
        help='define for which modality task should be performed')
    aa('--init', type=str, default=None,
        choices=[None, 'dSPoSE', 'normal'],
        help='initialisation of variational SPoSE model')
    aa('--data', type=str, default='human',
        choices=['dspose_init', 'human', 'synthetic'],
        help='define whether to use synthetically created triplet choices or human choices')
    aa('--reduction', type=str, default='sum',
        choices=['sum', 'max', 'l1_norm'],
        help='function applied to aggregate KL divergences across dimensions')
    aa('--results_dir', type=str, default='./results',
        help='optional specification of results directory (if not provided will resort to ./results/)')
    aa('--plot_dir', type=str, default='./plots',
        help='optional specification of directory for plots (if not provided will resort to ./plots/)')
    aa('--n_items', type=int, default=1854,
        help='number of unique items in dataset')
    aa('--top_j', type=int, default=30,
        help='top j dimensions in VSPoSE to visualize')
    aa('--top_k', type=int, default=20,
        help='top k items for each of the top j dimensions in VSPoSE to show')
    aa('--img_folder', type=str, default='./reference_images',
        help='provide folder name from where to load reference image for visualizing dimensions')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def correlations_to_txt(
                        plots_dir:str,
                        W_vspose:np.ndarray,
                        W_dspose:np.ndarray,
) -> None:
    #write correlations to .txt file
    with open(os.path.join(plots_dir, 'vspose_vs_dspose_dimensions.txt'), 'w', encoding='utf8') as f:
        for k, vspose_d in enumerate(W_vspose.T):
            rhos, p_vals = zip(*[pearsonr(vspose_d, dspose_d) for dspose_d in W_dspose.T])
            rhos_sorted = np.argsort(rhos)[::-1]
            rho_max_ind = rhos_sorted[0]
            rho_max = rhos[rho_max_ind]
            f.write('\n')
            f.write('=========================================================\n')
            f.write('\n')
            f.write(f'VSPoSE dimension: {k}; dSPoSE dimension: {rho_max_ind}\n')
            f.write('\n')
            f.write(f"Pearson Correlation: {rho_max:.3f}\n")
            f.write('\n')
            f.write('=========================================================')

def compute_pcs(
                W_vspose:np.ndarray,
                W_dspose:np.ndarray,
                n_components,
                rnd_seed:int,
) -> Tuple[list, list]:
    vpca = PCA(n_components=n_components, svd_solver='full', random_state=rnd_seed)
    dpca = PCA(n_components=n_components, svd_solver='full', random_state=rnd_seed)
    #transform data
    W_vspose_PCs = vpca.fit_transform(W_vspose)
    W_dspose_PCs = dpca.fit_transform(W_dspose)
    #return PCs and corresponding mapping matrix (to correspond to orig. dimension)
    H_PCs = [W_vspose_PCs, W_dspose_PCs]
    W_PCs = [vpca.components_, dpca.components_]
    return H_PCs, W_PCs

def visualize_dimensions(
                          results_dir:str,
                          plot_dir:str,
                          modality:str,
                          init:str,
                          data:str,
                          reduction:str,
                          n_items:int,
                          img_folder:str,
                          top_j:int,
                          top_k:int,
                          device:torch.device,
                          rnd_seed:int,
) -> None:
    #set variables
    dimensions = np.arange(100, 300, 100)
    lambdas = np.array([[5.5e+3, 6.5e+3, 7.5e+3], [1.3e+4, 1.4e+4, 1.5e+4, 1.6e+4]])

    #load item names, sort indices, and reference images into memory
    item_names, sortindex = utils.load_inds_and_item_names()
    ref_images = utils.load_ref_images(img_folder, item_names)
    assert len(item_names) == len(ref_images), '\nNumber of images must be equal to the number of item names.\n'

    for d, dim in enumerate(dimensions):
        #initialise deterministic SPoSE model
        dspose = SPoSE(in_size=n_items, out_size=dim)
        #load weights of converged dSPoSE model
        dspose = utils.load_model(
                            model=dspose,
                            results_dir=results_dir,
                            modality=modality,
                            version='deterministic',
                            data='human',
                            dim=dim,
                            lmbda=0.008,
                            rnd_seed=rnd_seed,
                            device=device,
        )

        #sort dSPoSE's weights and find top_k (k = 50) dimensions
        _, W_dspose = utils.sort_weights(dspose, aggregate=False)
        W_dspose = W_dspose[:, :50]
        W_dspose = W_dspose[sortindex]

        for lmbda in lambdas[d]:
            #initialise a variational version of SPoSE
            vspose = VSPoSE(in_size=n_items, out_size=dim, init_weights=True, init_method=init, device=device, rnd_seed=rnd_seed)
            #load weights of converged VSPoSE model with dSPoSE initialisation
            vspose = utils.load_model(
                                model=vspose,
                                results_dir=results_dir,
                                modality=modality,
                                version='variational',
                                data=data,
                                dim=dim,
                                lmbda=lmbda,
                                rnd_seed=rnd_seed,
                                device=device,
            )
            #sort VSPoSE dimensions in descending order according to their KLDs and find the top_k dims
            sorted_dims, _ = utils.compute_kld(vspose, lmbda=lmbda, aggregate=True, reduction=reduction)
            W_mu, _ = utils.load_weights(vspose, 'variational')
            W_mu = W_mu.numpy()
            W_mu = W_mu[sortindex]
            W_vspose = np.abs(W_mu) #take the abs transform W_mu into R+
            W_vspose = W_vspose[:, sorted_dims]

            #for each VSPoSE dimension, correlate against each of the top_k dSPoSE dimensions and find best fit
            plots_dir = os.path.join(plot_dir, modality, 'variational', data, f'{dim}d', f'{lmbda}', f'seed{rnd_seed:02d}')

            plot_top_k_dim_correlations(W_vspose=W_vspose[:, :50], W_dspose=W_dspose, plots_dir=plots_dir, top_k=50)
            correlations_to_txt(plots_dir, W_vspose, W_dspose)

            H_PCs, W_PCs = compute_pcs(W_vspose, W_dspose, n_components=2, rnd_seed=rnd_seed)
            plot_vspose_along_dpose_pcs(plots_dir=plots_dir, H_PCs=H_PCs, W_PCs=W_PCs)

            H_PCs, W_PCs = compute_pcs(W_vspose, W_dspose, n_components=.9, rnd_seed=rnd_seed)
            plot_vspose_against_dpose_pcs(plots_dir=plots_dir, H_PCs=H_PCs, W_PCs=W_PCs, n_rows=3, n_cols=3)

            #plot reference images wrt top k items for each of the top j dimensions in VSPoSE
            W_sorted = np.copy(W_vspose).T
            if top_j > 40:
                #split dim visualizations into two separate parts
                W_sorted_a = W_sorted[:int(top_j/2)]
                W_sorted_b = W_sorted[int(top_j/2):top_j]
                plot_imgs_along_dimensions(
                                            W_sorted=W_sorted_a,
                                            sorted_dims=sorted_dims,
                                            images=ref_images,
                                            top_j=top_j,
                                            top_k=top_k,
                                            plots_dir=plots_dir,
                                            part=1,
                )
                plot_imgs_along_dimensions(
                                            W_sorted=W_sorted_b,
                                            sorted_dims=sorted_dims,
                                            images=ref_images,
                                            top_j=top_j,
                                            top_k=top_k,
                                            plots_dir=plots_dir,
                                            part=2,
                )
            else:
                W_sorted = W_sorted[:top_j]
                plot_imgs_along_dimensions(
                                            W_sorted=W_sorted,
                                            sorted_dims=sorted_dims,
                                            images=ref_images,
                                            top_j=top_j,
                                            top_k=top_k,
                                            plots_dir=plots_dir,
                )

            dspose_dims_sorted, vspose_dims_sorted, corrs = utils.compare_modalities(W_dspose, W_vspose, duplicates=True)

            print('\n...Finished plotting individual dimensions for VSPoSE. Now visualizing DSPoSE and VSPoSE Dimensions alongside each other.\n')
            W_sorted_dspose = W_dspose[:, dspose_dims_sorted].T
            W_sorted_vspose = W_vspose[:, vspose_dims_sorted].T

            for j, (w_dspose, w_vspose) in enumerate(zip(W_sorted_dspose, W_sorted_vspose)):
                #move coefficients onto same scale to find most dissimilar objects between model versions
                w_dspose /= w_dspose.max()
                w_vspose /= w_vspose.max()
                #visualize top k objects for each model and k most dissimilar objects between models
                visualize_dims_across_modalities(
                                                plots_dir=plots_dir,
                                                ref_images=ref_images,
                                                corrs=corrs,
                                                w_mod1=w_dspose,
                                                w_mod2=w_vspose,
                                                latent_dim=j,
                                                difference='absolute',
                                                )

if __name__ == '__main__':
    #parse all arguments
    args = parseargs()
    #set random seeds (important for reproducibility)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    #set device
    device = torch.device(args.device)

    visualize_dimensions(
                            results_dir=args.results_dir,
                            plot_dir=args.plot_dir,
                            modality=args.modality,
                            init=args.init,
                            data=args.data,
                            reduction=args.reduction,
                            n_items=args.n_items,
                            top_j=args.top_j,
                            top_k=args.top_k,
                            img_folder=args.img_folder,
                            device=device,
                            rnd_seed=args.rnd_seed,
                            )
