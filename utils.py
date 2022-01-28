#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import pickle
import re
import torch

import numpy as np
import pandas as pd
import skimage.io as io
import torch.nn.functional as F

from collections import defaultdict
from itertools import islice
from os.path import join as pjoin
from skimage.transform import resize
from typing import Tuple, Iterator, List, Dict


class DataLoader(object):

    def __init__(
        self,
        I: torch.tensor,
        dataset: torch.Tensor,
        batch_size: int,
        train: bool=True,
    ):
        self.I = I
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        self.n_batches = math.ceil(len(self.dataset) / self.batch_size)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_batches(self.I, self.dataset)

    def get_batches(self, I: torch.Tensor, triplets: torch.Tensor) -> Iterator[torch.Tensor]:
        if self.train:
            triplets = triplets[torch.randperm(triplets.shape[0])]
        for i in range(self.n_batches):
            batch = encode_as_onehot(
                I, triplets[i * self.batch_size: (i + 1) * self.batch_size])
            yield batch


def pickle_file(file: dict, out_path: str, file_name: str) -> None:
    with open(os.path.join(out_path, ''.join((file_name, '.txt'))), 'wb') as f:
        f.write(pickle.dumps(file))


def unpickle_file(in_path: str, file_name: str) -> dict:
    return pickle.loads(open(os.path.join(in_path, ''.join((file_name, '.txt'))), 'rb').read())


def remove_nans(E: np.ndarray) -> np.ndarray:
    E_cp = E[:, :]
    # return indices for rows that contain NaN values
    nan_indices = np.isnan(E_cp).any(axis=1)
    E_cp = E_cp[~nan_indices]
    return E_cp


def assert_nneg(X: np.ndarray, thresh: float = 1e-5) -> np.ndarray:
    """if data matrix X contains negative real numbers, transform matrix into R+ (i.e., positive real number(s) space)"""
    if np.any(X < 0):
        X -= np.amin(X, axis=0)
        return X + thresh
    return X


def filter_triplets(rnd_samples: np.ndarray, n_samples: float) -> np.ndarray:
    """filter for unique triplets (i, j, k have to be different indices)"""
    rnd_samples = np.asarray(list(filter(lambda triplet: len(
        np.unique(triplet)) == len(triplet), rnd_samples)))
    # remove all duplicates from our sample
    rnd_samples = np.unique(rnd_samples, axis=0)[:int(n_samples)]
    return rnd_samples



def load_inds_and_item_names(folder: str = './data') -> Tuple[np.ndarray]:
    item_names = pd.read_csv(
        pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values
    sortindex = pd.read_table(
        pjoin(folder, 'sortindex'), header=None)[0].values
    return item_names, sortindex


def load_ref_images(img_folder: str, item_names: np.ndarray) -> np.ndarray:
    ref_images = np.array([resize(io.imread(pjoin('./reference_images', name + '.jpg')),
                          (400, 400), anti_aliasing=True) for name in item_names])
    return ref_images


def load_data(device: torch.device, triplets_dir: str, val_set: str = 'test_10', inference: bool = False) -> Tuple[torch.Tensor]:
    """load train and test triplet datasets into memory"""
    if inference:
        with open(pjoin(triplets_dir, 'test_triplets.npy'), 'rb') as test_triplets:
            test_triplets = torch.from_numpy(np.load(test_triplets)).to(
                device).type(torch.LongTensor)
            return test_triplets
    try:
        with open(pjoin(triplets_dir, 'train_90.npy'), 'rb') as train_file:
            train_triplets = torch.from_numpy(
                np.load(train_file)).to(device).type(torch.LongTensor)

        with open(pjoin(triplets_dir, f'{val_set}.npy'), 'rb') as test_file:
            test_triplets = torch.from_numpy(np.load(test_file)).to(
                device).type(torch.LongTensor)
    except FileNotFoundError:
        print('\n...Could not find any .npy files for current modality.')
        print('...Now searching for .txt files.\n')
        train_triplets = torch.from_numpy(np.loadtxt(
            pjoin(triplets_dir, 'train_90.txt'))).to(device).type(torch.LongTensor)
        test_triplets = torch.from_numpy(np.loadtxt(
            pjoin(triplets_dir, f'{val_set}.txt'))).to(device).type(torch.LongTensor)
    return train_triplets, test_triplets


def get_nitems(train_triplets: torch.Tensor) -> int:
    # number of unique items in the data matrix
    n_items = torch.max(train_triplets).item()
    if torch.min(train_triplets).item() == 0:
        n_items += 1
    return n_items


def load_batches(
    train_triplets: torch.Tensor,
    test_triplets: torch.Tensor,
    n_items: int,
    batch_size: int,
    inference: bool = False,
):
    # initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    if inference:
        assert train_triplets is None
        test_batches = DataLoader(
            I=I, dataset=test_triplets, batch_size=batch_size, train=False)
        return test_batches
    else:
        # create two iterators of train and validation mini-batches respectively
        train_batches = DataLoader(
            I=I, dataset=train_triplets, batch_size=batch_size, train=True)
        val_batches = DataLoader(
            I=I, dataset=test_triplets, batch_size=batch_size, train=False)
    return train_batches, val_batches


def encode_as_onehot(I: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    """encode item triplets as one-hot-vectors"""
    return I[triplets.flatten(), :]


################################################
######### helper functions for evaluation ######
################################################

def instance_sampling(probas: np.ndarray) -> np.ndarray:
    rnd_sample = np.random.choice(
        np.arange(len(probas)), size=len(probas), replace=True)
    probas_draw = probas[rnd_sample]
    return probas_draw


def get_global_averages(avg_probas: dict) -> np.ndarray:
    sorted_bins = dict(sorted(avg_probas.items()))
    return np.array([np.mean(p) for p in sorted_bins.values()])


def compute_pm(probas: np.ndarray) -> Tuple[np.ndarray, dict]:
    """compute probability mass for every choice"""
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


def mse(avg_p: np.ndarray, confidences: np.ndarray) -> float:
    return np.mean((avg_p - confidences)**2)


def bootstrap_calibrations(PATH: str, alpha: float, n_bootstraps: int = 1000) -> np.ndarray:
    mses = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        probas = seed_sampling(PATH)
        probas_draw = instance_sampling(probas)
        confidences, avg_p = compute_pm(probas_draw, alpha)
        mses[i] += mse(avg_p, confidences)
    return mses


def get_model_confidence_(PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    seeds = get_seeds(PATH)
    confidence_scores = np.zeros((len(seeds), 11))
    avg_probas = np.zeros((len(seeds), 11))
    for i, seed in enumerate(seeds):
        with open(os.path.join(PATH, seed, 'test_probas.npy'), 'rb') as f:
            confidence, avg_p = compute_pm(np.load(f))
            confidence_scores[i] += confidence
            avg_probas[i] += avg_p
    return confidence_scores, avg_probas


def mat2py(triplet: tuple) -> tuple:
    return tuple(np.asarray(triplet) - 1)


def pmf(hist: dict) -> np.ndarray:
    values = np.array(list(hist.values()))
    return values / np.sum(values)


def histogram(choices: list) -> dict:
    hist = {i + 1: 0 for i in range(3)}
    for choice in choices:
        hist[choice] += 1
    return hist


def compute_pmfs(choices: dict, behavior: bool) -> Dict[Tuple[int, int, int], np.ndarray]:
    if behavior:
        pmfs = {mat2py(t): pmf(histogram(c)) for t, c in choices.items()}
    else:
        pmfs = {t: np.array(pmfs).mean(axis=0) for t, pmfs in choices.items()}
    return pmfs


def get_choice_distributions(test_set: pd.DataFrame) -> dict:
    """function to compute human choice distributions and corresponding pmfs"""
    triplets = test_set[['trip.1', 'trip.2', 'trip.3']]
    test_set['triplets'] = list(map(tuple, triplets.to_numpy()))
    unique_triplets = test_set.triplets.unique()
    choice_distribution = defaultdict(list)
    for triplet in unique_triplets:
        choices = list(test_set[test_set['triplets'] == triplet].choice.values)
        sorted_choices = [
            np.where(np.argsort(triplet) + 1 == c)[0][0] + 1 for c in choices]
        sorted_triplet = tuple(sorted(triplet))
        choice_distribution[sorted_triplet].extend(sorted_choices)
    choice_pmfs = compute_pmfs(choice_distribution, behavior=True)
    return choice_pmfs



def collect_choices(probas:np.ndarray, human_choices:np.ndarray, model_choices:dict) -> dict:
    """collect model choices at inference time"""
    probas = probas.flip(dims=[1])
    for pmf, choices in zip(probas, human_choices):
        sorted_choices = tuple(np.sort(choices))
        model_choices[sorted_choices].append(pmf[np.argsort(choices)].numpy().tolist())
    return model_choices
    

def load_model(
    model,
    PATH: str,
    device: torch.device,
    subfolder: str = 'model',
):
    model_path = pjoin(PATH, subfolder)
    models = sorted(os.listdir(model_path))
    PATH = pjoin(model_path, models[-1])
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def pearsonr(u: np.ndarray, v: np.ndarray, a_min: float = -1., a_max: float = 1.) -> np.ndarray:
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)
    return rho


def robustness(corrs: np.ndarray, thresh: float) -> float:
    return len(corrs[corrs > thresh]) / len(corrs)
