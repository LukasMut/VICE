#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re

from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from typing import List

def plot_latent_causes(
                        plots_dir: str,
                        latent_causes: List[int],
                        show_plot: bool=False,
) -> None:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = plt.subplot(111)
    
    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.plot(np.arange(len(latent_causes)), latent_causes, alpha=0.8, linestyle='dashed')
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'Number of latent causes')
    
    PATH = os.path.join(plots_dir, 'latent_causes')
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    plt.savefig(os.path.join(PATH, 'latent_causes_over_time.png'))
    if show_plot:
        plt.show()

def plot_single_performance(
                            plots_dir:str,
                            val_accs:list,
                            train_accs:list,
                            steps: int,
                            show_plot: bool=False,
                            ) -> None:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.plot(val_accs,'-+',  alpha=.5, label='Test')
    ax.plot(train_accs[steps-1::steps], '-+', alpha=.5, label='Train')
    ax.set_xticks(ticks=range(len(val_accs)))
    ax.set_xticklabels(labels=list(range(steps, len(train_accs)+1, steps)))
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'Accuracy')
    ax.legend(fancybox=True, shadow=True, loc='best')

    PATH = pjoin(plots_dir, 'grid_search')
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    plt.savefig(pjoin(PATH, 'single_model_performance_over_time.png'))
    if show_plot:
        plt.show()
    
    plt.close()


def plot_complexities_and_loglikelihoods(
                                         plots_dir:str,
                                         loglikelihoods:list,
                                         complexity_losses:list,
                                         show_plot: bool=False,
                                         ) -> None:
    losses = [loglikelihoods, complexity_losses]
    labels = [r'$L^{E}$', r'$L^{C}$']
    ylabels = [r'Cross-entropy loss', r'Complexity cost']
    n_cols = len(losses)
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 10), dpi=100)

    for i, ax in enumerate(axes):
        #hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        #only show ticks on the left (y-axis) and bottom (x-axis) spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.plot(losses[i],'-o', alpha=.5, label=labels[i])
        ax.set_xlim([0, len(losses[i])])
        ax.set_xlabel(r'Epochs')
        ax.set_ylabel(ylabels[i])
        ax.legend(fancybox=True, shadow=True, loc='upper right')

    PATH = pjoin(plots_dir, 'losses')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, 'llikelihood_and_complexity_over_time.png'))
    if show_plot:
        plt.show()
    plt.close()


def plot_imgs_along_dimensions(
                                W_sorted:np.ndarray,
                                sorted_dims:np.ndarray,
                                images:np.ndarray,
                                top_j:int,
                                top_k:int,
                                plots_dir:str,
                                part=None,
                                show_plot:bool=False,
) -> None:
    """for each latent dimension sort coefficients in descending order, and plot top k objects"""
    #initialise figure
    n_rows = int(top_j/2) if top_j > 40 else top_j
    fig, axes = plt.subplots(n_rows, top_k, figsize=(50, 90), dpi=80)
    for j, w_j in enumerate(W_sorted):
        w_j_sorted = np.argsort(w_j) #ascending
        top_k_indices = w_j_sorted[::-1][:top_k] #descending
        top_k_images = images[top_k_indices]
        top_k_weights = w_j[top_k_indices]

        for k, (img, w) in enumerate(zip(top_k_images, top_k_weights)):
            axes[j, k].imshow(img)
            axes[j, k].set_xlabel(f'Weight: {w:.2f}')
            axes[j, k].set_xticks([])
            axes[j, k].set_yticks([])

        if part == 2:
            axes[j, 0].set_ylabel(f'{j+1+n_rows:02d}', fontsize=30, rotation=0, labelpad=30)
        else:
            axes[j, 0].set_ylabel(f'{j+1:02d}', fontsize=30, rotation=0, labelpad=30)

    PATH = pjoin(plots_dir, 'dim_visualizations')
    if not os.path.exists(PATH):
        print('\n...Creating directories.\n')
        os.makedirs(PATH)

    if isinstance(part, int):
        plt.savefig(pjoin(PATH, f'top_{top_k:02d}_objects_for_top_{top_j:02d}_dimensions_part_{part:01d}.jpg'))
    else:
        plt.savefig(pjoin(PATH, f'top_{top_k:02d}_objects_for_top_{top_j:02d}_dimensions.jpg'))
    if show_plot:
        plt.show()
    plt.close()

def plot_rsm(rsm:np.ndarray, plots_dir:str) -> None:
    PATH = pjoin(plots_dir, 'rsa')
    if not os.path.exists(PATH):
        print('\n...Creating directories.\n')
        os.makedirs(PATH)

    plt.figure(figsize=(10, 6), dpi=100)
    #plt.imshow(rsm, cmap=plt.cm.inferno)
    plt.imshow(rankdata(rsm).reshape(rsm.shape))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(pjoin(PATH, 'rsm.jpg'))
    plt.close()


def get_img_pairs(tril_inds:tuple, most_dissim:np.ndarray, ref_images:np.ndarray) -> np.ndarray:
    tril_inds_i = tril_inds[0][most_dissim]
    tril_inds_j = tril_inds[1][most_dissim]
    ref_images_i = ref_images[tril_inds_i]
    ref_images_j = ref_images[tril_inds_j]
    img_pairs = np.concatenate((np.concatenate(ref_images_i, axis=1), np.concatenate(ref_images_j, axis=1)), axis=0)
    return img_pairs
