#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import torch
import train

import numpy as np

os.environ['PYTHONIOENCODING'] = 'UTF-8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--modality', type=str, default='behavioral',
        help='define current modality (e.g., behavioral, visual, neural, text)')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/modality/init_dim/optim/prior/seed/spike/slab/pi)')
    aa('--plots_dir', type=str, default='./plots/',
        help='optional specification of directory for plots (if not provided will resort to ./plots/modality/init_dim/optim/prior/seed/spike/slab/pi)')
    aa('--epochs', metavar='T', type=int, default=2000,
        help='maximum number of epochs to run VICE optimization')
    aa('--burnin', type=int, default=500,
        help='minimum number of epochs to run VICE optimization')
    aa('--eta', type=float, default=0.001,
        help='learning rate to be used in optimizer')
    aa('--init_dim', metavar='D', type=int, default=100,
        help='initial dimensionality of the latent space')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets sampled during each step (i.e., mini-batch size)')
    aa('--optim', type=str, default='adam',
        choices=['adam', 'adamw', 'sgd'],
        help='optimizer to train VICE')
    aa('--prior', type=str, metavar='p', default='gaussian',
        choices=['gaussian', 'laplace'],
        help='whether to use a Gaussian or Laplacian mixture for the spike-and-slab prior')
    aa('--mc_samples', type=int, default=10,
        help='number of weight samples to use for MC sampling')
    aa('--spike', type=float, default=0.25,
        help='sigma for spike distribution')
    aa('--slab', type=float, default=1.0,
        help='sigma for slab distribution (should be smaller than spike)')
    aa('--pi', type=float, default=0.5,
        help='scalar value that determines the relative weight of the spike and slab distributions respectively')
    aa('--k', type=int, default=5,
        choices=[5, 10],
        help='minimum number of items that have non-zero weight for a latent dimension (according to importance scores)')
    aa('--ws', type=int, default=500,
        help='determines for how many epochs the number of latent dimensions (after pruning) is not allowed to vary')
    aa('--steps', type=int, default=50,
        help='perform validation and save model parameters every <steps> epochs')
    aa('--device', type=str, default='cpu',
        help='whether training should be performed on CPU or GPU (i.e., CUDA).')
    aa('--num_threads', type=int, default=4,
        help='number of threads used for intraop parallelism on CPU; use only if device is CPU')
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility of results')
    aa('--verbose', action='store_true',
        help='whether to display print statements about model performance during training')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    if re.search(r'cuda', args.device):
        device = torch.device(args.device)
        torch.cuda.manual_seed_all(args.rnd_seed)
        try:
            current_device = int(args.device[-1])
        except ValueError:
            current_device = 1
        try:
            torch.cuda.set_device(current_device)
        except RuntimeError:
            current_device = 0
            torch.cuda.set_device(current_device)
        device = torch.device(f'cuda:{current_device}')
        print(f'\nPyTorch CUDA version: {torch.version.cuda}')
        print(f'Process is running on *cuda:{current_device}*\n')
    else:
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
        torch.set_num_threads(args.num_threads)
        device = torch.device(args.device)

    train.run(
        modality=args.modality,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        triplets_dir=args.triplets_dir,
        epochs=args.epochs,
        burnin=args.burnin,
        eta=args.eta,
        batch_size=args.batch_size,
        init_dim=args.init_dim,
        optim=args.optim,
        prior=args.prior,
        mc_samples=args.mc_samples,
        spike=args.spike,
        slab=args.slab,
        pi=args.pi,
        k=args.k,
        ws=args.ws,
        steps=args.steps,
        device=device,
        rnd_seed=args.rnd_seed,
        verbose=args.verbose,
    )