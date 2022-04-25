#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re

from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from typing import List

def plot_latent_dimensions(
                        plots_dir: str,
                        latent_dimensions: List[int],
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

    ax.plot(np.arange(len(latent_dimensions)), latent_dimensions, alpha=0.8, linestyle='dashed')
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'Number of latent causes')
    
    PATH = os.path.join(plots_dir, 'latent_dimensions')
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        
    plt.savefig(os.path.join(PATH, 'latent_dimensions_over_time.png'))
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


def plot_topk_objects_per_dimension(
                                    plots_dir: str,
                                    images: np.ndarray,
                                    w_j: np.ndarray,
                                    latent_dim: int,
                                    top_k: int=6,
                                    show_plot: bool=True,
) -> None:
    topk_objects = np.argsort(-w_j)[:top_k]
    topk_images = images[topk_objects]

    def concat_imgs(images:np.ndarray, top_k:int) -> np.ndarray:
        img_combination = np.concatenate([
            np.concatenate([img for img in images[:int(top_k/2)]], axis = 1),
            np.concatenate([img for img in images[int(top_k/2):]], axis = 1)], axis = 0)
        return img_combination

    img_name = f'vice_laten_dim_{latent_dim:02d}.png'
    border_col = 'black'
    img_comb = concat_imgs(images=topk_images, top_k=top_k)

    #set variables and initialise figure object
    fig = plt.figure(figsize=(14, 4), dpi=150)
    ax = plt.subplot(111)
    
    for spine in ax.spines:
        ax.spines[spine].set_color(border_col)
        ax.spines[spine].set_linewidth(2.25)
    
    ax.imshow(img_comb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(f'Dimension {latent_dim+1:02d}', labelpad=15, fontsize=15)

    PATH = os.path.join(plots_dir, 'interpretability')
    if not os.path.exists(PATH):
        print('\n...Creating directories.\n')
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, img_name), bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
