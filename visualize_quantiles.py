#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import utils

from skimage.transform import resize
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io


def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--out_path', type=str,
        help='path/to/plots')
    aa('--weights', type=str,
        help='path/to/median/model/weights')
    aa('--image_folder', type=str,
        help='path/to/image/files')
    aa('--n_quants', type=int,
        help='number of quantiles')
    aa('--thresh', type=float,
        help='threshold for nonzero weights')
    aa('--weight_scale', action='store_true',
        help='whether or not to plot a weight scale above image quantiles')
    aa('--scale_file', type=str, default=None,
        help='path/to/weight/scale')
    args = parser.parse_args()
    return args


def get_quantiles(w: np.ndarray, n_quants: int):
    return np.quantile(w, np.linspace(0, 1, n_quants))


def get_sample(subset: np.ndarray, n: int):
    return np.random.choice(subset, size=n, replace=False)


def get_subsets(w: np.ndarray, n_quants: int, n_objects: int, thresh: float) -> List[np.ndarray]:
    quantiles = get_quantiles(w[np.where(w > thresh)[0]], n_quants)[::-1]
    zeros = np.where(w == 0.)[0]
    subsets = []
    for i in range(n_quants - 1):
        subset = np.where((w < quantiles[i]) & (w > quantiles[i + 1]))[0]
        sample = get_sample(subset, n_objects[i])
        subsets.append(sample)
    zero_sample = get_sample(zeros, n_objects[0])
    subsets.append(zero_sample)
    return subsets


def concat_images(images: np.ndarray, indices: np.ndarray) -> np.ndarray:
    img_combination = np.concatenate([
        np.concatenate([img for img in images[:int(indices/2)]], axis = 0),
        np.concatenate([img for img in images[int(indices/2):]], axis = 0)], axis = 1)
    return img_combination


def concat_image_subsets(images: np.ndarray, subsets: List[np.ndarray]) -> np.ndarray:
    img_combs = []
    for i, subset in enumerate(subsets):
        object_quantile = images[subset]
        if len(subset) == 6:
            img_comb = concat_images(object_quantile, len(subset))
            placeholder = np.ones_like(np.concatenate(object_quantile[:len(subset)//2], axis=0))
        else:
            img_comb = np.concatenate(object_quantile, axis=0)
            placeholder = np.ones_like(img_comb)
        img_combs.append(img_comb)
        if i < len(subsets) - 1:
            img_combs.append(placeholder)
    img_combs = np.concatenate(img_combs, axis=1)
    return img_combs
        
            
def plot_dim_quantiles(
                        images: np.ndarray,
                        subsets: np.ndarray,
                        latent_dim: int,
                        out_path: str,
                        show_plot: bool=True,
                        weight_scale: bool=False,
                        scale_file=None,
) -> None:
    img_comb = concat_image_subsets(images, subsets)
    if weight_scale:
        assert isinstance(scale_file, str)
        img_name = f'vice_quantiles_scale_laten_dim_{latent_dim:02d}.jpg'
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), dpi=500)
        for i, ax in enumerate(axes):
            if i == 0:
                scale = io.imread(scale_file)
                scale = resize(scale, (img_comb.shape[0] // 2, img_comb.shape[1]), anti_aliasing=False)
                ax.imshow(scale)
            else:
                ax.imshow(img_comb)
        
            for spine in ax.spines:
                ax.spines[spine].set_color('white')
                ax.spines[spine].set_linewidth(0.1)
    else:
        img_name = f'vice_quantiles_laten_dim_{latent_dim:02d}.jpg'
        fig = plt.figure(figsize=(14, 4), dpi=500)
        ax = plt.subplot(111)
        ax.imshow(img_comb)
        for spine in ax.spines:
            ax.spines[spine].set_color('white')
            ax.spines[spine].set_linewidth(0.1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    PATH = os.path.join(plots_dir, 'interpretability')
    if not os.path.exists(PATH):
        print('\n...Creating directories.\n')
        os.makedirs(PATH)

    plt.savefig(os.path.join(PATH, img_name))
    if show_plot:
        plt.show()
    plt.close()


def visualize_quantiles(
                        W: np.ndarray,
                        images: np.ndarray,
                        n_quants: int,
                        n_objects: int,
                        thresh: float,
                        out_path: str,
                        weight_scale: bool=False,
                        scale_file=None,
) -> None:
    for j, w in enumerate(W):
        subsets = get_subsets(w, n_quants, n_objects, thresh)
        plot_dim_quantiles(
                            images=images,
                            subsets=subsets,
                            latent_dim=j,
                            out_path=out_path,
                            show_plot=True,
                            weight_scale=weight_scale,
                            scale_file=scale_file,
        )


if __name__ == '__main__':
    args = parseargs()
    item_names, _ = utils.load_inds_and_item_names()
    images = utils.load_ref_images(args.image_folder, item_names)
    W = np.load(args.weights)
    n_objects = [6 if i == 0 else 3 for i in range(args.n_quants)]
    visualize_quantiles(
                        W=W,
                        images=images,
                        n_quants=args.n_quants,
                        n_objects=n_objects,
                        thresh=args.thresh,
                        out_path=args.out_path,
                        weight_scale=args.weight_scale,
                        scale_file=args.scale_file,
    )