#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple


def get_results(in_path: str, thresh: float = .9,
) -> List[Tuple[float, float, float, float]]:
    results = []
    for root, _, files in os.walk(in_path):
        for f in files:
            if f == 'results_1000.json':
                root_list = root.split('/')
                spike = float(root_list[-3])
                slab = float(root_list[-2])
                pi = float(root_list[-1])
                if (spike < slab and slab < 2**3):
                    with open(os.path.join(root, f), 'r') as json_file:
                        r = json.load(json_file)
                        c_entropy = r['val_loss']
                        if not np.isnan(c_entropy):
                            if c_entropy < thresh:
                                results.append((spike, slab, pi, c_entropy))
    return results


def plot_error_surface(
    results: List[Tuple[float, float, float, float]],
    out_path: str,
) -> None:
    spikes, slabs, pis, losses = zip(*results)

    fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={"projection": "3d"})

    ax.set_xlabel(r'$\sigma_{1}$')
    ax.set_ylabel(r'$\sigma_{2}$')
    ax.set_zlabel(r'$\pi$', rotation=0)
    ax = ax.scatter(np.log(spikes), np.log(slabs), pis,
                    c=losses, cmap='viridis', linewidth=0.3)
    fig.colorbar(ax, shrink=0.7, aspect=10)
    plt.tight_layout()
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(os.path.join(out_path, 'error_surface.png'))
    plt.close()


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    thresh = float(sys.argv[3])

    results = get_results(in_path=in_path, thresh=thresh)
    plot_error_surface(results=results, out_path=out_path)
