#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import torch
import utils
import numpy as np
import torch.nn.functional as F

from models.model import SPoSE
from typing import List, Iterator

os.environ['PYTHONIOENCODING']='UTF-8'

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--n_samples', type=int, default=1,
        help='define how many different synthetic triplet datasets you would like to sample')
    aa('--version', type=str, default='deterministic',
        choices=['deterministic', 'variational'],
        help='whether to apply a deterministic or variational version of SPoSE')
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--modality', type=str, default='behavioral/',
        help='define for which modality SPoSE should be perform specified task')
    aa('--triplets_dir', type=str, default='./triplets',
        help='in case you have tripletized data, provide directory from where to load triplets')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/modality/version/dim/lambda/rnd_seed/)')
    aa('--embed_dim', metavar='D', type=int, default=90,
        help='dimensionality of the embedding matrix (i.e., out_size of model)')
    aa('--batch_size', metavar='B', type=int, default=100,
        choices=[16, 25, 32, 50, 64, 100, 128, 150, 200, 256],
        help='number of triplets in each mini-batch')
    aa('--lmbda', type=float,
        help='lambda value determines scale of l1 regularization')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def sampling(model, val_batches:Iterator[torch.Tensor], task:str, batch_size:int, device:torch.device) -> np.ndarray:
    sampled_choices = np.zeros((int(len(val_batches) * batch_size), 3), dtype=int)
    model.eval()
    with torch.no_grad():
        for j, batch in enumerate(val_batches):
            batch = batch.to(device)
            logits = model(batch)
            anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)
            similarities = utils.compute_similarities(anchor, positive, negative, task)
            probas = F.softmax(torch.stack(similarities, dim=-1), dim=1).numpy()
            probas = probas[:, ::-1]
            human_choices = batch.nonzero(as_tuple=True)[-1].view(batch_size, -1).numpy()
            model_choices = np.array([np.random.choice(h_choice, size=len(p), replace=False, p=p)[::-1] for h_choice, p in zip(human_choices, probas)])
            sampled_choices[j*batch_size:(j+1)*batch_size] += model_choices
    return sampled_choices

def run(
        n_samples:int,
        version:str,
        task:str,
        modality:str,
        results_dir:str,
        triplets_dir:str,
        lmbda:float,
        batch_size:int,
        embed_dim:int,
        rnd_seed:int,
        device:torch.device,
) -> None:
    #load train triplets
    train_triplets, test_triplets = utils.load_data(device=device, triplets_dir=os.path.join(triplets_dir, modality))
    n_items = utils.get_nitems(train_triplets)
    train_batches, val_batches = utils.load_batches(train_triplets=train_triplets, test_triplets=test_triplets, n_items=n_items, batch_size=batch_size, rnd_seed=rnd_seed)
    for i in range(n_samples):
        #create directories
        PATH = os.path.join(triplets_dir, 'synthetic', f'sample_{i+1:02d}')
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        #initialise model
        model = SPoSE(in_size=n_items, out_size=embed_dim)
        model = utils.load_model(
                                   model=model,
                                   results_dir=results_dir,
                                   modality=modality,
                                   version=version,
                                   data='human',
                                   dim=embed_dim,
                                   lmbda=lmbda,
                                   rnd_seed=rnd_seed,
                                   device=device,
                                   )
        model.to(device)
        #probabilistically sample triplet choices given model ouput PMFs (train set)
        sampled_choices = sampling(model, train_batches, task, batch_size, device)
        with open(os.path.join(PATH, 'train_90.npy'), 'wb') as f:
            np.save(f, sampled_choices)
        #probabilistically sample triplet choices given model ouput PMFs (val set)
        sampled_choices = sampling(model, val_batches, task, batch_size, device)
        with open(os.path.join(PATH, 'test_10.npy'), 'wb') as f:
            np.save(f, sampled_choices)

        print(f'\nFinished sampling synthetic dataset No. {i}\n')

if __name__ == '__main__':
    #parse all arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    #set device
    device = torch.device(args.device)
    if args.device != 'cpu':
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        try:
            torch.cuda.set_device(int(args.device[-1]))
        except:
            torch.cuda.set_device(1)
        print(f'\nPyTorch CUDA version: {torch.version.cuda}\n')

    run(
        n_samples=args.n_samples,
        version=args.version,
        task=args.task,
        modality=args.modality,
        results_dir=args.results_dir,
        triplets_dir=args.triplets_dir,
        lmbda=args.lmbda,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        rnd_seed=args.rnd_seed,
        device=args.device,
        )
