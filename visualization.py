#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import torch

from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, rankdata
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
    # ax.annotate('Val acc: {:.3f}'.format(np.max(val_accs)), (len(val_accs) - len(val_accs) * 0.1, np.max(val_accs) / 2))
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

def plot_multiple_performances(
                                plots_dir:str,
                                val_accs:list,
                                train_accs:list,
                                lambdas:np.ndarray,
) -> None:
    n_rows = len(lambdas) // 2
    n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=100)
    max_conv = max(list(map(lambda accs: len(accs), val_accs)))

    #keep track of k
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            #hide the right and top spines
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['top'].set_visible(False)

            #only show ticks on the left (y-axis) and bottom (x-axis) spines
            axes[i, j].yaxis.set_ticks_position('left')
            axes[i, j].xaxis.set_ticks_position('bottom')

            axes[i, j].plot(val_accs[k],'-+',  alpha=.5, label='Test')
            axes[i, j].plot(train_accs[k], '-+', alpha=.5, label='Train')
            axes[i, j].annotate('Val acc: {:.3f}'.format(np.max(val_accs)), (max_conv - max_conv * 0.1, np.max(val_accs) / 2))
            axes[i, j].set_xlim([0, max_conv])
            axes[i, j].set_xlabel(r'Epochs')
            axes[i, j].set_ylabel(r'Accuracy')
            axes[i, j].set_title(f'Lambda-L1: {lambdas[k]}')
            axes[i, j].legend(fancybox=True, shadow=True, loc='lower left')
            k += 1

    for ax in axes.flat:
        ax.label_outer()

    PATH = pjoin(plots_dir, 'grid_search')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, 'model_performances_over_time.png'))
    plt.close()

def plot_val_accs_across_seeds(plots_dir:str, lmbdas:np.ndarray, val_accs:np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 8), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.plot(lmbdas, val_accs*100)
    ax.set_xticks(lmbdas)
    ax.set_xlabel(f'$\lambda')
    ax.set_ylabel(r'Val acc (%)')

    plt.savefig(pjoin(plots_dir, 'lambda_search_results.png'))
    plt.close()

def plot_grid_search_results(
                            results:dict,
                            plot_dir:str,
                            rnd_seed:int,
                            modality:str,
                            version:str,
                            subfolder:str,
                            vision_model=None,
                            layer=None,
) -> None:
    fig = plt.figure(figsize=(16, 8), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    lambdas = list(map(lambda l: round(float(l), 4), results.keys()))
    train_accs, val_accs = zip(*[(val['train_acc'], val['val_acc']) for lam, val in results.items()])

    ax.plot(train_accs, alpha=.8, label='Train')
    ax.plot(val_accs, alpha=.8, label='Val')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(lambdas)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel(r'$\lambda$')
    ax.legend(fancybox=True, shadow=True, loc='upper right')
    plt.tight_layout()

    if modality == 'visual':
        assert isinstance(vision_model, str) and isinstance(layer, str), 'name of vision model and corresponding layer are required'
        PATH = pjoin(plot_dir, f'seed{rnd_seed}', modality, vision_model, layer, version, subfolder)
    else:
        PATH = pjoin(plot_dir, f'seed{rnd_seed}', modality, version, subfolder)
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, 'lambda_search_results.png'))
    plt.close()

def plot_dim_correlations(
                          W_mu_vspose:torch.Tensor,
                          W_mu_dspose:torch.Tensor,
                          plots_dir:str,
                          epoch:int,
                          ) -> None:
    """Pearson correlations between top k VSPoSE and dSPoSE dimensions"""
    fig = plt.figure(figsize=(16, 8), dpi=200)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    top_k = 50
    rhos = np.array([pearsonr(dspose_d, vspose_d)[0] for dspose_d, vspose_d in zip(W_mu_dspose[:, :top_k].T, W_mu_vspose[:, :top_k].T)])
    ax.bar(np.arange(len(rhos)), rhos, alpha=.5)
    ax.set_ylabel(r'$\rho$', fontsize=13)
    ax.set_xlabel('Dimension', fontsize=13)
    ax.set_title(f'Epoch: {epoch}', fontsize=13)

    PATH = pjoin(plots_dir, 'dim_correlations')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, f'dim_correlations_{epoch:03d}.png'))
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


def plot_weight_violins(
                        W_sorted:np.ndarray,
                        plot_dir:str,
                        rnd_seed:int,
                        modality:str,
                        version:str,
                        data:str,
                        dim:int,
                        lmbda:float,
                        reduction:str,
                        ) -> None:
    """violinplot of KL divergences (VSPoSE) or l1-norms (SPoSE) across all items and latent dimensions"""
    fig = plt.figure(figsize=(16, 8), dpi=200)
    ax = plt.subplot(111)

    #y-axis label is dependent on version of SPoSE
    y_lab = 'KLD' if version == 'variational' else 'L1'

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.violinplot(W_sorted, widths=0.8)
    ax.set_xticks(np.arange(0, W_sorted.shape[1]+1, 10))
    ax.set_xlabel('Dimension', fontsize=10)
    ax.set_ylabel(y_lab, fontsize=10)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)

    PATH = pjoin(plot_dir, modality, version, data, f'{dim}d', f'{lmbda}', f'seed{rnd_seed:02d}')

    if version == 'variational':
        PATH = pjoin(PATH, 'klds')
    else:
        PATH = pjoin(PATH, 'l1_norms')

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, f'dim_violinplot_{reduction}.png'))
    plt.close()

def plot_pruning_results(
                         results:list,
                         plot_dir:str,
                         rnd_seed:int,
                         modality:str,
                         version:str,
                         data:str,
                         dim:int,
                         lmbda:float,
                         reduction:str,
                         ) -> None:
    """plot validation accuracy as a function of pruned weights percentage"""
    fig = plt.figure(figsize=(16, 8), dpi=100)
    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    pruning_fracs, val_accs = zip(*results)
    ax.bar(pruning_fracs, val_accs, alpha=.5, width=4.0)
    ax.set_xticks(pruning_fracs)
    ax.set_xticklabels(pruning_fracs)
    ax.set_ylim([np.floor(np.min(val_accs)), np.ceil(np.max(val_accs))])
    ax.set_ylabel('Val acc (%)')
    ax.set_xlabel(r'% of weights pruned')

    PATH = pjoin(plot_dir, modality, version, data, f'{dim}d', f'{lmbda}', f'seed{rnd_seed:02d}', 'pruning')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, f'val_acc_against_pruned_weights_{reduction}.png'))
    plt.close()

def plot_top_k_dim_correlations(
                                W_vspose:np.ndarray,
                                W_dspose:np.ndarray,
                                plots_dir:str,
                                top_k:int,
) -> None:
    K = W_vspose.shape[1]
    n_rows = 10
    n_cols = K // n_rows

    #set font size and spacing variables
    label_size = 8
    title_size = 15

    #adjust alpha according to Bonferroni correction (to account for multiple comparisons problem)
    alpha_corrected = .05 / K

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=200)
    fig.suptitle(f'Top {top_k} VSPoSE ($x$-axis) vs. dSPoSE ($y$-axis) dimensions', y=0.925, fontsize=title_size)

    #keep track of k
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):

            rhos, p_vals = zip(*[pearsonr(W_vspose[:, k], dspose_d)[0] for dspose_d in W_dspose.T])
            rhos_sorted = np.argsort(rhos)[::-1]
            rho_max_ind = rhos_sorted[0]
            rho_max = rhos[rho_max_ind]
            p_val = p_vals[rho_max_ind]
            x = W_vspose[:, k]
            y = W_dspose[:, rho_max_ind]

            rho_str = str(round(rho_max, 3))
            #r_text = r'$\rho = \mathbf{' + rho_str + '}$' if p_val < alpha_corrected else
            r_text = r'$\rho = ' + rho_str + '$'

            #hide the right and top spines
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['top'].set_visible(False)

            #only show ticks on the left (y-axis) and bottom (x-axis) spines
            axes[i, j].yaxis.set_ticks_position('left')
            axes[i, j].xaxis.set_ticks_position('bottom')

            #plot correlation
            axes[i, j].scatter(x, y, alpha=.5)

            #annotate correlation values (i.e., rho-values)
            axes[i, j].annotate(r_text, (np.min(x), np.max(y)), fontsize=label_size+1)

            axes[i, j].set_xlabel(f'Dim {k+1}', fontsize=label_size)
            axes[i, j].set_ylabel(f'Dim {rho_max_ind+1}', fontsize=label_size)
            #remove x- and y-axis ticks
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            k += 1

    PATH = pjoin(plots_dir, 'dim_correlations')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, f'top_k_vspose_vs_dspose_dimensions.png'))
    plt.close()

def plot_vspose_against_dpose_pcs(
                                    plots_dir:str,
                                    H_PCs:np.ndarray,
                                    W_PCs:np.ndarray,
                                    n_rows:int,
                                    n_cols:int,
) -> None:
    #initialize figure object
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), dpi=100)

    #keep track of running index k to index PCs
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            #hide the right and top spines
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['top'].set_visible(False)

            #only show ticks on the left (y-axis) and bottom (x-axis) spines
            axes[i, j].yaxis.set_ticks_position('left')
            axes[i, j].xaxis.set_ticks_position('bottom')

            #plot top PCs of each model against each other
            axes[i, j].scatter(H_PCs[0][:, k], H_PCs[1][:, k], c=H_PCs[0][:, k]+H_PCs[1][:, k], cmap=plt.cm.magma)

            #find dimensions that are associated with top PCs
            vpc_dim = np.argmax(W_PCs[0][k])
            dpc_dim = np.argmax(W_PCs[1][k])

            axes[i, j].set_xlabel(f'V-SPoSE PC{k+1} ($\cong$ Dim {vpc_dim})')
            axes[i, j].set_ylabel(f'D-SPoSE PC{k+1} ($\cong$ Dim {dpc_dim})')

            #remove x- and y-axes tickes
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            k += 1

    PATH = pjoin(plots_dir, 'first_principal_components')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, 'vspose_against_dspose_pcs.png'))
    plt.close()

def plot_vspose_along_dpose_pcs(
                                plots_dir:str,
                                H_PCs:np.ndarray,
                                W_PCs:np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=100)
    titles = ['V-SPoSE', 'D-SPoSE']
    for i, ax in enumerate(axes):
        #hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        #only show ticks on the left (y-axis) and bottom (x-axis) spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #for each model, plot top PCs against each other
        ax.scatter(H_PCs[i][:, 0], H_PCs[i][:, 1], c=H_PCs[i][:, 0]+H_PCs[i][:, 1], cmap=plt.cm.magma)

        #in feature space, find dimensions that are associated with top PCs
        pc1_dim = np.argmax(W_PCs[i][0])
        pc2_dim = np.argmax(W_PCs[i][1])

        ax.set_xlabel(f'PC1 ($\cong$ Dim {pc1_dim})', fontsize=12)
        ax.set_ylabel(f'PC2 ($\cong$ Dim {pc2_dim})', fontsize=12)
        ax.set_title(titles[i], fontsize=13)

    PATH = pjoin(plots_dir, 'first_principal_components')
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, 'vspose_and_dspose_pcs.png'))
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

def visualize_dims_across_modalities(
                                    plots_dir:str,
                                    ref_images:np.ndarray,
                                    corrs:np.ndarray,
                                    w_mod1:np.ndarray,
                                    w_mod2:np.ndarray,
                                    latent_dim:int,
                                    duplicates=None,
                                    results_path=None,
                                    searchlight=None,
                                    difference:str='absolute',
                                    top_k:int=8,
                                    show_plot:bool=False,
) -> None:
    from utils import get_image_combinations, load_searchlight_imgs
    #find both top k objects per modality and their intersection
    topk_mod1 = np.argsort(-w_mod1)[:top_k]
    topk_mod2 = np.argsort(-w_mod2)[:top_k]
    top_k_common = np.intersect1d(topk_mod1, topk_mod2)
    #find top k objects (i.e., objects with hightest loadings) in current latent dimension for each modality
    topk_imgs_mod1 = ref_images[topk_mod1]
    topk_imgs_mod2 = ref_images[topk_mod2]

    #calculate rank or absolute differences of weight coefficients between the (two) modalities
    if difference == 'rank':
        rank_diff_mod1 = rankdata(-w_mod1)-rankdata(-w_mod2)
        rank_diff_mod2 = rankdata(-w_mod2)-rankdata(-w_mod1)
        most_dissim_imgs_mod1 = ref_images[np.argsort(rank_diff_mod1)[:top_k]]
        most_dissim_imgs_mod2 = ref_images[np.argsort(rank_diff_mod2)[:top_k]]
    else:
        abs_diff_mod1 = w_mod1 - w_mod2
        abs_diff_mod2 = w_mod2 - w_mod1
        most_dissim_imgs_mod1 = ref_images[np.argsort(abs_diff_mod1)[::-1][:top_k]]
        most_dissim_imgs_mod2 = ref_images[np.argsort(abs_diff_mod2)[::-1][:top_k]]

    if re.search(r'variational', plots_dir):
        x_lab = 'D-SPoSE'
        y_lab = 'V-SPoSE'
        img_name = f'dspose_vspose_viz_laten_dim_{latent_dim:02d}.jpg'
        titles = [r'D-SPoSE', r'V-SPoSE']
        border_cols = ['r', 'b']
        imgs_combs = get_image_combinations(
                                            topk_imgs_mod1=topk_imgs_mod1,
                                            topk_imgs_mod2=topk_imgs_mod2,
                                            most_dissim_imgs_mod1=most_dissim_imgs_mod1,
                                            most_dissim_imgs_mod2=most_dissim_imgs_mod2,
                                            top_k=top_k,
                                            )
        n_rows = len(imgs_combs) // 2
        n_cols = n_rows + 1
    else:
        x_lab = 'Behavior'
        y_lab = 'VGG 16'
        img_name = f'mind_machine_viz_laten_dim_{latent_dim:02d}.jpg'
        titles = [r'Human Behavior', r'VGG 16', r'VGG 16']
        border_cols = ['r', 'b', 'b']

        PATH = pjoin(results_path, searchlight)
        PATH = pjoin(PATH, 'duplicates', f'{latent_dim:02d}') if duplicates else pjoin(PATH, 'no_duplicates', f'{latent_dim:02d}')
        PATH_topk = pjoin(PATH, 'top_k')
        PATH_mostdissim = pjoin(PATH, 'most_dissim')

        #with open(pjoin(PATH, 'img_identifiers.json')) as j_file:
        #    img_identifiers = json.load(j_file)

        #load searchlight images into memory
        topk_imgs_searchlight = load_searchlight_imgs(PATH_topk)
        most_dissim_imgs_searchlight = load_searchlight_imgs(PATH_mostdissim)
        imgs_combs = get_image_combinations(
                                            topk_imgs_mod1=topk_imgs_mod1,
                                            topk_imgs_mod2=topk_imgs_mod2,
                                            most_dissim_imgs_mod1=most_dissim_imgs_mod1,
                                            most_dissim_imgs_mod2=most_dissim_imgs_mod2,
                                            top_k=top_k,
                                            topk_imgs_searchlight=topk_imgs_searchlight,
                                            most_dissim_imgs_searchlight=most_dissim_imgs_searchlight,
                                            )
        n_rows = len(imgs_combs) // 3
        n_cols = n_rows * 2

    #set variables and initialise figure object
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (14, 4), dpi=300)
    y_labs = [r'Top $k$', r'Most dissimilar']
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if j < (n_cols - 1):
                for spine in axes[i, j].spines:
                    axes[i, j].spines[spine].set_color(border_cols[j])
                    axes[i, j].spines[spine].set_linewidth(1.75)

                if i == 0:
                    axes[i, j].set_title(titles[j])

                axes[i, j].imshow(imgs_combs[k])
                k += 1
            else:
                #for each scatter plot, reset all colors to grey before coloring respective images
                colors = np.array(['grey' for _ in range(len(w_mod1))])

                #hide the right and top spines
                axes[i, j].spines['right'].set_visible(False)
                axes[i, j].spines['top'].set_visible(False)
                #only show ticks on the left (y-axis) and bottom (x-axis) spines
                axes[i, j].yaxis.set_ticks_position('left')
                axes[i, j].xaxis.set_ticks_position('bottom')

                if i == 0:
                    colors[topk_mod1] = 'r'
                    colors[topk_mod2] = 'b'
                    if len(top_k_common) > 0:
                        colors[top_k_common] = 'm'
                else:
                    if difference == 'rank':
                        colors[np.argsort(rank_diff_mod1)[:top_k]] = 'r'
                        colors[np.argsort(rank_diff_mod2)[:top_k]] = 'b'
                    else:
                        colors[np.argsort(abs_diff_mod1)[::-1][:top_k]] = 'r'
                        colors[np.argsort(abs_diff_mod2)[::-1][:top_k]] = 'b'

                axes[i, j].scatter(w_mod1, w_mod2, c=colors, alpha=.6)
                axes[i, j].set_xlabel(x_lab)
                axes[i, j].set_ylabel(y_lab)

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

        #axes[i, 0].set_ylabel(' '.join((r'$\rho$','=',f'{corrs[latent_dim]:.3f}')))
        axes[i, 0].set_ylabel(y_labs[i])

    PATH = pjoin(plots_dir, 'dim_comparison')
    if not os.path.exists(PATH):
        print('\n...Creating directories.\n')
        os.makedirs(PATH)

    plt.savefig(pjoin(PATH, img_name))
    if show_plot:
        plt.show()
    plt.close()


def get_img_pairs(tril_inds:tuple, most_dissim:np.ndarray, ref_images:np.ndarray) -> np.ndarray:
    tril_inds_i = tril_inds[0][most_dissim]
    tril_inds_j = tril_inds[1][most_dissim]
    ref_images_i = ref_images[tril_inds_i]
    ref_images_j = ref_images[tril_inds_j]
    img_pairs = np.concatenate((np.concatenate(ref_images_i, axis=1), np.concatenate(ref_images_j, axis=1)), axis=0)
    return img_pairs
