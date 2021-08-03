#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import math
import os
import pickle
import re
import torch
import warnings

import numpy as np
import pandas as pd
import skimage.io as io
import torch.nn.functional as F

from collections import defaultdict, Counter
from itertools import islice, combinations, permutations
from numba import njit, jit, prange
from os.path import join as pjoin
from skimage.transform import resize
from torch.distributions.normal import Normal
from torch.optim import Adam, AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import Tuple, Iterator, List, Dict

class BatchGenerator(object):

    def __init__(
                self,
                I:torch.tensor,
                dataset:torch.Tensor,
                batch_size:int,
                sampling_method:str='normal',
                p=None,
):
        self.I = I
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampling_method = sampling_method
        self.p = p

        if sampling_method == 'soft':
            assert isinstance(self.p, float)
            self.n_batches = int(len(self.dataset) * self.p) // self.batch_size
        else:
            self.n_batches = len(self.dataset) // self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_batches(self.I, self.dataset)

    def sampling(self, triplets:torch.Tensor) -> torch.Tensor:
        """randomly sample training data during each epoch"""
        rnd_perm = torch.randperm(len(triplets))
        if self.sampling_method == 'soft':
            rnd_perm = rnd_perm[:int(len(rnd_perm) * self.p)]
        return triplets[rnd_perm]

    def get_batches(self, I:torch.Tensor, triplets:torch.Tensor) -> Iterator[torch.Tensor]:
        if not isinstance(self.sampling_method, type(None)):
            triplets = self.sampling(triplets)
        for i in range(self.n_batches):
            batch = encode_as_onehot(I, triplets[i*self.batch_size: (i+1)*self.batch_size])
            yield batch

def pickle_file(file:dict, out_path:str, file_name:str) -> None:
    with open(os.path.join(out_path, ''.join((file_name, '.txt'))), 'wb') as f:
        f.write(pickle.dumps(file))

def unpickle_file(in_path:str, file_name:str) -> dict:
    return pickle.loads(open(os.path.join(in_path, ''.join((file_name, '.txt'))), 'rb').read())

def remove_nans(E:np.ndarray) -> np.ndarray:
    E_cp = E[:, :]
    nan_indices = np.isnan(E_cp).any(axis=1) #return indices for rows that contain NaN values
    E_cp = E_cp[~nan_indices]
    return E_cp

def assert_nneg(X:np.ndarray, thresh:float=1e-5) -> np.ndarray:
    """if data matrix X contains negative real numbers, transform matrix into R+ (i.e., positive real number(s) space)"""
    if np.any(X < 0):
        X -= np.amin(X, axis=0)
        return X + thresh
    return X

def load_features(PATH:str) -> np.ndarray:
    if re.search(r'text', PATH):
        E = np.loadtxt(PATH, delimiter=',')
        E = remove_nans(E) #remove all objects that contain NaN values
    else:
        with open(PATH, 'rb') as f:
            E = np.load(f)
    return E

def filter_triplets(rnd_samples:np.ndarray, n_samples:float) -> np.ndarray:
    """filter for unique triplets (i, j, k have to be different indices)"""
    rnd_samples = np.asarray(list(filter(lambda triplet: len(np.unique(triplet)) == len(triplet), rnd_samples)))
    #remove all duplicates from our sample
    rnd_samples = np.unique(rnd_samples, axis=0)[:int(n_samples)]
    return rnd_samples

def tripletize_data(PATH:str, method:str, n_samples:float, sampling_constant:float, folder:str, dir:str='./triplets', device:torch.device=torch.device('cpu')) -> Tuple[np.ndarray]:
    """create triplets of object embedding similarities, and for each triplet find the odd-one-out"""
    #load word embeddings or DNN hidden unit activations into memory
    E = load_features(PATH)
    #compute similarity matrix
    #TODO: figure out whether an affinity matrix might be more reasonable (i.e., informative) than a simple similarity matrix
    S = matul(E, E.T)
    N = S.shape[0]
    #draw random samples of triplets of concepts
    rnd_samples = np.random.randint(N, size=(int(n_samples + sampling_constant), 3))
    #filter for unique triplets and remove all duplicates
    rnd_samples = filter_triplets(rnd_samples, n_samples)

    triplets = np.zeros((int(n_samples), 3), dtype=int)
    for idx, [i, j, k] in enumerate(rnd_samples):
        odd_one_outs = np.asarray([k, j, i])
        sims = np.array([S[i, j], S[i, k], S[j, k]])
        choices = odd_one_outs[np.argsort(sims)]
        triplets[idx] = choices

    PATH = pjoin(dir, folder)
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    rnd_indices = np.random.permutation(len(triplets))
    train_triplets = triplets[rnd_indices[:int(len(rnd_indices)*.9)]]
    test_triplets = triplets[rnd_indices[int(len(rnd_indices)*.9):]]

    with open(pjoin(PATH, 'train_90.npy'), 'wb') as train_file:
        np.save(train_file, train_triplets)

    with open(pjoin(PATH, 'test_10.npy'), 'wb') as test_file:
        np.save(test_file, test_triplets)

    train_triplets = torch.from_numpy(train_triplets).to(device).type(torch.LongTensor)
    test_triplets = torch.from_numpy(test_triplets).to(device).type(torch.LongTensor)

    return train_triplets, test_triplets

def load_inds_and_item_names(folder:str='./data') -> Tuple[np.ndarray]:
    item_names = pd.read_csv(pjoin(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values
    sortindex = pd.read_table(pjoin(folder, 'sortindex'), header=None)[0].values
    return item_names, sortindex

def load_ref_images(img_folder:str, item_names:np.ndarray) -> np.ndarray:
    ref_images = np.array([resize(io.imread(pjoin('./reference_images', name + '.jpg')), (400, 400), anti_aliasing=True) for name in item_names])
    return ref_images

def load_concepts(folder:str='./data') -> pd.DataFrame:
    concepts = pd.read_csv(pjoin(folder, 'category_mat_manual.tsv'), encoding='utf-8', sep='\t')
    return concepts

def load_data(device:torch.device, triplets_dir:str, inference:bool=False) -> Tuple[torch.Tensor]:
    """load train and test triplet datasets into memory"""
    if inference:
        with open(pjoin(triplets_dir, 'test_triplets.npy'), 'rb') as test_triplets:
            test_triplets = torch.from_numpy(np.load(test_triplets)).to(device).type(torch.LongTensor)
            return test_triplets
    try:
        with open(pjoin(triplets_dir, 'train_90.npy'), 'rb') as train_file:
            train_triplets = torch.from_numpy(np.load(train_file)).to(device).type(torch.LongTensor)

        with open(pjoin(triplets_dir, 'test_10.npy'), 'rb') as test_file:
            test_triplets = torch.from_numpy(np.load(test_file)).to(device).type(torch.LongTensor)
    except FileNotFoundError:
        print('\n...Could not find any .npy files for current modality.')
        print('...Now searching for .txt files.\n')
        train_triplets = torch.from_numpy(np.loadtxt(pjoin(triplets_dir, 'train_90.txt'))).to(device).type(torch.LongTensor)
        test_triplets = torch.from_numpy(np.loadtxt(pjoin(triplets_dir, 'test_10.txt'))).to(device).type(torch.LongTensor)
    return train_triplets, test_triplets

def get_nitems(train_triplets:torch.Tensor) -> int:
    #number of unique items in the data matrix
    n_items = torch.max(train_triplets).item()
    if torch.min(train_triplets).item() == 0:
        n_items += 1
    return n_items

def load_batches(
                 train_triplets:torch.Tensor,
                 test_triplets:torch.Tensor,
                 n_items:int,
                 batch_size:int,
                 inference:bool=False,
                 sampling_method:str=None,
                 rnd_seed:int=None,
                 p=None,
                 ):
    #initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    if inference:
        assert train_triplets is None
        test_batches = BatchGenerator(I=I, dataset=test_triplets, batch_size=batch_size, sampling_method=None, p=None)
        return test_batches
    else:
        #create two iterators of train and validation mini-batches respectively
        train_batches = BatchGenerator(I=I, dataset=train_triplets, batch_size=batch_size, sampling_method=sampling_method, p=p)
        val_batches = BatchGenerator(I=I, dataset=test_triplets, batch_size=batch_size, sampling_method=None, p=None)
    return train_batches, val_batches

def pdf(sample:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor):
    return torch.exp(-((sample - mu) ** 2) / (2 * sigma.pow(2))) / sigma * math.sqrt(2 * math.pi)

def log_pdf(sample:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor) -> torch.Tensor:
    return -((sample - mu) ** 2) / (2 * sigma.pow(2)) - (sigma.log() + math.log(math.sqrt(2 * math.pi)))

def spike_and_slab(sample:torch.Tensor, mu:torch.Tensor, sigma_1:torch.Tensor, sigma_2:torch.Tensor, pi:float) -> torch.Tensor:
    assert pi < 1, 'the relative weight pi is required to be < 1'
    spike = pi * pdf(sample, mu, sigma_1)
    slab = (1 - pi) * pdf(sample, mu, sigma_2)
    return spike + slab

def encode_as_onehot(I:torch.Tensor, triplets:torch.Tensor) -> torch.Tensor:
    """encode item triplets as one-hot-vectors"""
    return I[triplets.flatten(), :]

def softmax(sims:tuple, t:torch.Tensor) -> torch.Tensor:
    return torch.exp(sims[0] / t) / torch.sum(torch.stack([torch.exp(sim / t) for sim in sims]), dim=0)

def cross_entropy_loss(sims:tuple, t:torch.Tensor) -> torch.Tensor:
    return torch.mean(-torch.log(softmax(sims, t)))

def compute_similarities(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str) -> Tuple:
    pos_sim = torch.sum(anchor * positive, dim=1)
    neg_sim = torch.sum(anchor * negative, dim=1)
    if method == 'odd_one_out':
        neg_sim_2 = torch.sum(positive * negative, dim=1)
        return pos_sim, neg_sim, neg_sim_2
    else:
        return pos_sim, neg_sim

def accuracy_(probas:np.ndarray) -> float:
    choices = np.where(probas.mean(axis=1) == probas.max(axis=1), -1, np.argmax(probas, axis=1))
    acc = np.where(choices == 0, 1, 0).mean()
    return acc

def choice_accuracy(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str) -> float:
    similarities  = compute_similarities(anchor, positive, negative, method)
    probas = F.softmax(torch.stack(similarities, dim=-1), dim=1).detach().cpu().numpy()
    return accuracy_(probas)

def trinomial_probs(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str, t:torch.Tensor) -> torch.Tensor:
    sims = compute_similarities(anchor, positive, negative, method)
    return softmax(sims, t)

def trinomial_loss(anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, method:str, t:torch.Tensor) -> torch.Tensor:
    sims = compute_similarities(anchor, positive, negative, method)
    return cross_entropy_loss(sims, t)

def kld_online(mu_1:torch.Tensor, l_1:torch.Tensor, mu_2:torch.Tensor, l_2:torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.log(l_1/l_2) + (l_2/l_1) * torch.exp(-l_1 * torch.abs(mu_1-mu_2)) + l_2*torch.abs(mu_1-mu_2) - 1)

def kld_offline(mu_1:torch.Tensor, b_1:torch.Tensor, mu_2:torch.Tensor, b_2:torch.Tensor) -> torch.Tensor:
    return torch.log(b_2/b_1) + (b_1/b_2) * torch.exp(-torch.abs(mu_1-mu_2)/b_1) + torch.abs(mu_1-mu_2)/b_2 - 1

def get_nneg_dims(W:torch.Tensor, eps:float=0.1) -> int:
    w_max = W.max(dim=1)[0]
    nneg_d = len(w_max[w_max > eps])
    return nneg_d

################################################
######### helper functions for evaluation ######
################################################

def get_seeds(PATH:str) -> List[str]:
    return [dir.name for dir in os.scandir(PATH) if dir.is_dir() and dir.name.startswith('seed')]

def seed_sampling(PATH:str) -> np.ndarray:
    seed = np.random.choice(get_seeds(PATH))
    with open(os.path.join(PATH, seed, 'test_probas.npy'), 'rb') as f:
        probas = np.load(f)
    return probas

def instance_sampling(probas:np.ndarray) -> np.ndarray:
    rnd_sample = np.random.choice(np.arange(len(probas)), size=len(probas), replace=True)
    probas_draw = probas[rnd_sample]
    return probas_draw

def get_global_averages(avg_probas:dict) -> np.ndarray:
    sorted_bins = dict(sorted(avg_probas.items()))
    return np.array([np.mean(p) for p in sorted_bins.values()])

def compute_pm(probas:np.ndarray) -> Tuple[np.ndarray, dict]:
    """compute probability mass for every choice"""
    avg_probas = defaultdict(list)
    count_vector = np.zeros((2, 11))
    for pmf in probas:
        indices = np.round(pmf*10).astype(int)
        count_vector[0, indices[0]] += 1
        count_vector[1, indices] += 1
        for k, p in enumerate(pmf):
            avg_probas[int(indices[k])].append(p)
    model_confidences = count_vector[0]/count_vector[1]
    avg_probas = get_global_averages(avg_probas)
    return model_confidences, avg_probas

def mse(avg_p:np.ndarray, confidences:np.ndarray) -> float:
    return np.mean((avg_p - confidences)**2)

def bootstrap_calibrations(PATH:str, alpha:float, n_bootstraps:int=1000) -> np.ndarray:
    mses = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        probas = seed_sampling(PATH)
        probas_draw = instance_sampling(probas)
        confidences, avg_p = compute_pm(probas_draw, alpha)
        mses[i] += mse(avg_p, confidences)
    return mses

def get_model_confidence_(PATH:str) -> Tuple[np.ndarray, np.ndarray]:
    seeds = get_seeds(PATH)
    confidence_scores = np.zeros((len(seeds), 11))
    avg_probas = np.zeros((len(seeds), 11))
    for i, seed in enumerate(seeds):
        with open(os.path.join(PATH, seed, 'test_probas.npy'), 'rb') as f:
            confidence, avg_p = compute_pm(np.load(f))
            confidence_scores[i] += confidence
            avg_probas[i] += avg_p
    return confidence_scores, avg_probas

def mat2py(triplet:tuple) -> tuple:
    return tuple(np.asarray(triplet)-1)

def pmf(hist:dict) -> np.ndarray:
    values = np.array(list(hist.values()))
    return values/np.sum(values)

def histogram(choices:list) -> dict:
    hist = {i+1: 0 for i in range(3)}
    for choice in choices:
        hist[choice] += 1
    return hist

def compute_pmfs(choices:dict, behavior:bool) -> Dict[Tuple[int, int, int], np.ndarray]:
    if behavior:
        pmfs = {mat2py(t): pmf(histogram(c)) for t, c in choices.items()}
    else:
        pmfs = {t: np.array(pmfs).mean(axis=0) for t, pmfs in choices.items()}
    return pmfs

def get_choice_distributions(test_set:pd.DataFrame) -> dict:
    """function to compute human choice distributions and corresponding pmfs"""
    triplets = test_set[['trip.1', 'trip.2', 'trip.3']]
    test_set['triplets'] = list(map(tuple, triplets.to_numpy()))
    unique_triplets = test_set.triplets.unique()
    choice_distribution = defaultdict(list)
    for triplet in unique_triplets:
        choices = list(test_set[test_set['triplets']==triplet].choice.values)
        sorted_choices = [np.where(np.argsort(triplet)+1==c)[0][0]+1 for c in choices]
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

def mc_sampling(
                model,
                batch:torch.Tensor,
                task:str,
                n_samples:int,
                device:torch.device,
                temp:float=1.0,
                compute_stds:bool=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_alternatives = 3 if task == 'odd_one_out' else 2
    sampled_probas = torch.zeros(n_samples, batch.shape[0] // n_alternatives, n_alternatives).to(device)
    sampled_choices = torch.zeros(n_samples, batch.shape[0] // n_alternatives).to(device)

    for k in range(n_samples):
        logits, _, _, _ = model(batch)
        anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)
        similarities = compute_similarities(anchor, positive, negative, task)
        soft_choices = softmax(similarities, temp)
        probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
        sampled_probas[k] += probas
        sampled_choices[k] +=  soft_choices

    probas = sampled_probas.mean(dim=0)
    val_acc = accuracy_(probas.cpu().numpy())
    soft_choices = sampled_choices.mean(dim=0)
    val_loss = torch.mean(-torch.log(soft_choices))
    if compute_stds:
        stds = sampled_probas.std(dim=0)
        return val_acc, val_loss, probas, stds
    return val_acc, val_loss, probas

def test(
        model,
        test_batches,
        task:str,
        device:torch.device,
        n_samples:int,
        batch_size:int,
        temp:float=1.0,
        compute_stds:bool=False,
) -> Tuple:
    probas = torch.zeros(int(len(test_batches) * batch_size), 3)
    if compute_stds:
        triplet_stds = torch.zeros(int(len(test_batches) * batch_size), 3)
    model_choices = defaultdict(list)
    model.eval()
    with torch.no_grad():
        batch_accs = torch.zeros(len(test_batches))
        batch_centropies = torch.zeros(len(test_batches))
        for j, batch in enumerate(test_batches):
            batch = batch.to(device)
            if compute_stds:
                test_acc, test_loss, batch_probas, batch_stds = mc_sampling(
                                                                            model=model,
                                                                            batch=batch,
                                                                            temp=temp,
                                                                            task=task,
                                                                            n_samples=n_samples,
                                                                            device=device,
                                                                            compute_stds=compute_stds,
                                                                            )
                triplet_stds[j*batch_size:(j+1)*batch_size] += batch_stds
            else:
                test_acc, test_loss, batch_probas = mc_sampling(
                                                                model=model,
                                                                batch=batch,
                                                                temp=temp,
                                                                task=task,
                                                                n_samples=n_samples,
                                                                device=device,
                                                                )
            probas[j*batch_size:(j+1)*batch_size] += batch_probas
            batch_accs[j] += test_acc
            batch_centropies += test_loss
            human_choices = batch.nonzero(as_tuple=True)[-1].view(batch_size, -1).cpu().numpy()
            model_choices = collect_choices(batch_probas, human_choices, model_choices)

    probas = probas.cpu().numpy()
    probas = probas[np.where(probas.sum(axis=1) != 0.)]
    model_pmfs = compute_pmfs(model_choices, behavior=False)
    test_acc = batch_accs.mean().item()
    test_loss = batch_centropies.mean().item()
    if compute_stds:
        triplet_stds = triplet_stds.cpu().numpy()
        triplet_stds = triplet_stds.mean(axis=1)
        return test_acc, test_loss, probas, model_pmfs, triplet_stds
    return test_acc, test_loss, probas, model_pmfs

def validation(model, val_batches, task:str, device:torch.device, n_samples:int) -> Tuple[float, float]:
    temp = torch.tensor(1.).to(device)
    model.eval()
    with torch.no_grad():
        batch_losses_val = torch.zeros(len(val_batches))
        batch_accs_val = torch.zeros(len(val_batches))
        for j, batch in enumerate(val_batches):
            batch = batch.to(device)
            val_acc, val_loss, _ = mc_sampling(model=model, batch=batch, task=task, n_samples=n_samples, device=device, temp=temp)
            batch_losses_val[j] += val_loss.item()
            batch_accs_val[j] += val_acc
    avg_val_loss = torch.mean(batch_losses_val).item()
    avg_val_acc = torch.mean(batch_accs_val).item()
    return avg_val_loss, avg_val_acc

def get_results_files(
                      results_dir:str,
                      modality:str,
                      version:str,
                      subfolder:str,
                      vision_model=None,
                      layer=None,
) -> list:
    if modality == 'visual':
        assert isinstance(vision_model, str) and isinstance(layer, str), 'name of vision model and layer are required'
        PATH = pjoin(results_dir, modality, vision_model, layer, version, f'{dim}d', f'{lmbda}')
    else:
        PATH = pjoin(results_dir, modality, version, f'{dim}d', f'{lmbda}')
    files = [pjoin(PATH, seed, f) for seed in os.listdir(PATH) for f in os.listdir(pjoin(PATH, seed)) if f.endswith('.json')]
    return files

def sort_results(results:dict) -> dict:
    return dict(sorted(results.items(), key=lambda kv:kv[0], reverse=False))

def merge_dicts(files:list) -> dict:
    """merge multiple .json files efficiently into a single dictionary"""
    results = {}
    for f in files:
        with open(f, 'r') as f:
            results.update(dict(json.load(f)))
    results = sort_results(results)
    return results

def load_model(
                model,
                PATH:str,
                device:torch.device,
                subfolder:str='model',
):
    model_path = pjoin(PATH, subfolder)
    models = sorted(os.listdir(model_path))
    PATH = pjoin(model_path, models[-1])
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_weights_(out_path:str, W_mu:torch.tensor, W_b:torch.tensor) -> None:
    with open(pjoin(out_path, 'weights_mu_sorted.npy'), 'wb') as f:
        np.save(f, W_mu)
    with open(pjoin(out_path, 'weights_b_sorted.npy'), 'wb') as f:
        np.save(f, W_b)

def load_final_weights(out_path:str, version:str='variational') -> None:
    if version == 'variational':
        with open(pjoin(out_path, 'weights_mu_sorted.npy'), 'rb') as f:
            W_mu = np.load(f)
        with open(pjoin(out_path, 'weights_b_sorted.npy'), 'rb') as f:
            W_b = np.load(f)
        return W_mu, W_b
    else:
        with open(pjoin(out_path, 'weights_sorted.npy'), 'rb') as f:
            W = np.load(f)
        return W

def load_weights(model) -> Tuple[torch.Tensor, torch.Tensor]:
    W_mu = model.encoder_mu.weight.data.T.detach()
    #W_b = model.encoder_logb.weight.data.exp().T.detach()
    W_sigma = model.encoder_logsigma.weight.data.T.exp().detach()
    W_mu = F.relu(W_mu)
    return W_mu, W_sigma

def prune_weights(model, version:str, indices:torch.Tensor, fraction:float):
    indices = indices[:int(len(indices)*fraction)]
    for n, m in model.named_parameters():
        if version == 'variational':
            if re.search(r'encoder', n):
                #prune output weights and biases of encoder
                m.data = m.data[indices]
            else:
                #only prune input weights of decoder
                if re.search(r'weight', n):
                    m.data = m.data[:, indices]
        else:
            #prune output weights of fc layer
            m.data = m.data[indices]
    return model

def kld_cut_off(klds:np.ndarray) -> int:
    return np.argmax([(kld_i-kld_j) for kld_i, kld_j in zip(klds, islice(klds, 1, None))])

def compute_kld(model, lmbda:float, aggregate:bool, reduction=None) -> np.ndarray:
    mu_hat, b_hat = load_weights(model)
    mu = torch.zeros_like(mu_hat)
    lmbda = torch.tensor(lmbda)
    b = torch.ones_like(b_hat).mul(lmbda.pow(-1))
    kld = kld_offline(mu_hat, b_hat, mu, b)
    if aggregate:
        assert isinstance(reduction, str), '\noperator to aggregate KL divergences must be defined\n'
        if reduction == 'mean':
            #use sum as to aggregate KLDs for each dimension
            kld_mean = kld.mean(dim=0)
            sorted_dims = torch.argsort(kld_mean, descending=True)
            klds_sorted = kld_mean[sorted_dims].cpu().numpy()
        else:
            #use max to aggregate KLDs for each dimension
            kld_max = kld.max(dim=0)[0]
            sorted_dims = torch.argsort(kld_max, descending=True)
            klds_sorted = kld_max[sorted_dims].cpu().numpy()
    else:
        #use mean KLD to sort dimensions in descending order (highest KLDs first)
        sorted_dims = torch.argsort(kld.mean(dim=0), descending=True)
        klds_sorted = kld[:, sorted_dims].cpu().numpy()
    return sorted_dims, klds_sorted

#############################################################################################
######### helper functions to load weight matrices and compare RSMs across modalities #######
#############################################################################################

def robustness(corrs:np.ndarray, thresh:float) -> float:
    return len(corrs[corrs>thresh])/len(corrs)

def cross_correlate_latent_dims(X, thresh:float=None) -> float:
    if isinstance(X, np.ndarray):
        W_mu_i = np.copy(X)
        W_mu_j = np.copy(X)
    else:
        W_mu_i, W_mu_j = X
    corrs = np.zeros(min(W_mu_i.shape))
    for i, w_i in enumerate(W_mu_i):
        if np.all(W_mu_i == W_mu_j):
            corrs[i] = np.max([pearsonr(w_i, w_j) for j, w_j in enumerate(W_mu_j) if j != i])
        else:
            corrs[i] = np.max([pearsonr(w_i, w_j) for w_j in W_mu_j])
    if thresh:
        return robustness(corrs, thresh)
    return np.mean(corrs)
