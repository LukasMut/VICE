#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import skimage.io as io
import torch
from scipy.stats import norm
from skimage.transform import resize
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader

Tensor = torch.Tensor
Array = np.ndarray


def pickle_file(file: dict, out_path: str, file_name: str) -> None:
    with open(os.path.join(out_path, "".join((file_name, ".txt"))), "wb") as f:
        f.write(pickle.dumps(file))


def unpickle_file(in_path: str, file_name: str) -> dict:
    return pickle.loads(
        open(os.path.join(in_path, "".join((file_name, ".txt"))), "rb").read()
    )


def load_ref_images(img_folder: str, item_names: Array) -> Array:
    ref_images = np.array(
        [
            resize(
                io.imread(os.path.join(img_folder, name + ".jpg")),
                (400, 400),
                anti_aliasing=True,
            )
            for name in item_names
        ]
    )
    return ref_images


def load_data(
    device: torch.device,
    triplets_dir: str,
    val_set: str = "test_10",
    inference: bool = False,
) -> Tuple[Tensor]:
    """Load train and test triplet datasets from disk."""
    if inference:
        with open(
            os.path.join(triplets_dir, "test_triplets.npy"), "rb"
        ) as test_triplets:
            test_triplets = (
                torch.from_numpy(np.load(test_triplets))
                .to(device)
                .type(torch.LongTensor)
            )
            return test_triplets
    try:
        with open(os.path.join(triplets_dir, "train_90.npy"), "rb") as train_file:
            train_triplets = (
                torch.from_numpy(np.load(train_file)).to(device).type(torch.LongTensor)
            )

        with open(os.path.join(triplets_dir, f"{val_set}.npy"), "rb") as test_file:
            test_triplets = (
                torch.from_numpy(np.load(test_file)).to(device).type(torch.LongTensor)
            )
    except FileNotFoundError:
        print("\n...Could not find any .npy files for current modality.")
        print("...Now searching for .txt files.\n")
        train_triplets = (
            torch.from_numpy(np.loadtxt(os.path.join(triplets_dir, "train_90.txt")))
            .to(device)
            .type(torch.LongTensor)
        )
        test_triplets = (
            torch.from_numpy(np.loadtxt(os.path.join(triplets_dir, f"{val_set}.txt")))
            .to(device)
            .type(torch.LongTensor)
        )
    return train_triplets, test_triplets


def get_nobjects(train_triplets: Tensor) -> int:
    # number of unique items in the data matrix
    n_objects = torch.max(train_triplets).item()
    if torch.min(train_triplets).item() == 0:
        n_objects += 1
    return n_objects


def get_batches(triplets: Array, batch_size: int, train: bool) -> Iterator:
    dl = DataLoader(
        dataset=triplets,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=0,
        drop_last=False,
        pin_memory=True if train else False,
    )
    return dl


################################################
######### helper functions for evaluation ######
################################################


def instance_sampling(probas: Array) -> Array:
    rnd_sample = np.random.choice(
        np.arange(len(probas)), size=len(probas), replace=True
    )
    probas_draw = probas[rnd_sample]
    return probas_draw


def get_global_averages(avg_probas: dict) -> Array:
    sorted_bins = dict(sorted(avg_probas.items()))
    return np.array([np.mean(p) for p in sorted_bins.values()])


def compute_pm(probas: Array) -> Tuple[Array, dict]:
    """Compute the probability mass for every choice."""
    avg_probas = defaultdict(list)
    count_vector = np.zeros((2, 11))
    for pmf in probas:
        indices = np.round(pmf * 10).astype(int)
        count_vector[0, indices[0]] += 1
        count_vector[1, indices] += 1
        for k, p in enumerate(pmf):
            avg_probas[int(indices[k])].append(p)
    model_confidences = count_vector[0] / count_vector[1]
    avg_probas = get_global_averages(avg_probas)
    return model_confidences, avg_probas


def mse(avg_p: Array, confidences: Array) -> float:
    return np.mean((avg_p - confidences) ** 2)


def mat2py(triplet: tuple) -> tuple:
    return tuple(np.asarray(triplet) - 1)


def pmf(hist: dict) -> Array:
    values = np.array(list(hist.values()))
    return values / np.sum(values)


def histogram(choices: list) -> dict:
    hist = {i + 1: 0 for i in range(3)}
    for choice in choices:
        hist[choice] += 1
    return hist


def compute_pmfs(choices: dict, behavior: bool) -> Dict[Tuple[int, int, int], Array]:
    if behavior:
        pmfs = {mat2py(t): pmf(histogram(c)) for t, c in choices.items()}
    else:
        pmfs = {t: np.array(pmfs).mean(axis=0) for t, pmfs in choices.items()}
    return pmfs


def get_choice_distributions(test_set: pd.DataFrame) -> dict:
    """Compute human choice distributions and the corresponding PMFs."""
    triplets = test_set[["trip.1", "trip.2", "trip.3"]]
    test_set["triplets"] = list(map(tuple, triplets.to_numpy()))
    unique_triplets = test_set.triplets.unique()
    choice_distribution = defaultdict(list)
    for triplet in unique_triplets:
        choices = list(test_set[test_set["triplets"] == triplet].choice.values)
        sorted_choices = [
            np.where(np.argsort(triplet) + 1 == c)[0][0] + 1 for c in choices
        ]
        sorted_triplet = tuple(sorted(triplet))
        choice_distribution[sorted_triplet].extend(sorted_choices)
    choice_pmfs = compute_pmfs(choice_distribution, behavior=True)
    return choice_pmfs


def collect_choices(probas: Array, human_choices: Array, model_choices: dict) -> dict:
    """Collect model choices at inference time."""
    probas = probas.flip(dims=[1])
    for pmf, choices in zip(probas, human_choices):
        sorted_choices = tuple(np.sort(choices))
        model_choices[sorted_choices].append(pmf[np.argsort(choices)].numpy().tolist())
    return model_choices


def load_model(
    model,
    PATH: str,
    device: torch.device,
    subfolder: str = "model",
):
    model_path = os.path.join(PATH, subfolder)
    models = sorted(os.listdir(model_path))
    PATH = os.path.join(model_path, models[-1])
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def pearsonr(u: Array, v: Array, a_min: float = -1.0, a_max: float = 1.0) -> Array:
    """Compute the Pearson correlation coefficient."""
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)
    return rho


def robustness(corrs: Array, thresh: float) -> float:
    return len(corrs[corrs > thresh]) / len(corrs)


def compute_pvals(W_loc: Array, W_scale: Array) -> Array:
    # Compute the probability for an embedding value x_{ij} <= 0,
    # given mu and sigma of the variational posterior q_{\theta}
    def pval(W_loc, W_scale, j):
        return norm.cdf(0.0, W_loc[:, j], W_scale[:, j])

    return partial(pval, W_loc, W_scale)(np.arange(W_loc.shape[1])).T


def fdr_corrections(p_vals: Array, alpha: float = 0.05) -> Array:
    # For each dimension, statistically test how many objects have non-zero weight
    return np.array(
        list(map(lambda p: multipletests(p, alpha=alpha, method="fdr_bh")[0], p_vals))
    )


def get_importance(rejections: Array) -> Tuple[Array, Array]:
    # Yield the the number of rejections given by the False Discovery Rates
    return np.array(list(map(sum, rejections)))[:, np.newaxis]
