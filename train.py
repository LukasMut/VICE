#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
import utils
import visualization

import numpy as np

from models.model import VICE
from trainer import Trainer
from typing import Tuple

os.environ['PYTHONIOENCODING'] = 'UTF-8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# number of threads/CPU cores to be used per Python process (set to 2 if HT is enabled on CPU)
os.environ['OMP_NUM_THREADS'] = '1'


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--modality', type=str, default='behavioral',
        help='define current modality (e.g., behavioral, visual, neural, text)')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/modality/latent_dim/optim/prior/seed/spike/slab/pi)')
    aa('--plots_dir', type=str, default='./plots/',
        help='optional specification of directory for plots (if not provided will resort to ./plots/modality/latent_dim/optim/prior/seed/spike/slab/pi)')
    aa('--epochs', metavar='T', type=int, default=1000,
        help='maximum number of epochs to optimize SPoSE model for')
    aa('--eta', type=float, default=0.001,
        help='learning rate to be used in optimizer')
    aa('--latent_dim', metavar='D', type=int, default=100,
        help='dimensionality of the embedding matrix')
    aa('--batch_size', metavar='B', type=int, default=128,
        help='number of triplets subsampled during each iteration (i.e., mini-batch size)')
    aa('--optim', type=str, default='adam',
        choices=['adam', 'adamw', 'sgd'],
        help='optimizer to train VICE')
    aa('--prior', type=str, metavar='p', default='gaussian',
        choices=['gaussian', 'laplace'],
        help='whether to use a mixture of Gaussians or Laplacians for the spike-and-slab prior')
    aa('--mc_samples', type=int, default=20,
        choices=[10, 15, 20, 25, 30, 35, 40, 45, 50],
        help='number of weight samples to use for MC sampling')
    aa('--spike', type=float,
        help='sigma for spike distribution')
    aa('--slab', type=float,
        help='sigma for slab distribution')
    aa('--pi', type=float,
        help='scalar value that determines the relative weight of the spike and slab distributions respectively')
    aa('--steps', type=int, default=50,
        help='perform validation and save model parameters every <steps> epochs')
    aa('--device', type=str, default='cpu',
        help='whether training should be performed on CPU or GPU (i.e., CUDA).')
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility of results')
    aa('--verbose', action='store_true',
        help='whether to display print statements about model performance during training')
    args = parser.parse_args()
    return args


def create_dirs(
    results_dir: str,
    plots_dir: str,
    modality: str,
    latent_dim: int,
    optim: str,
    prior: str,
    spike: float,
    slab: float,
    pi: float,
    rnd_seed: int,
) -> Tuple[str, str, str]:
    """Create directories for results, plots, and model parameters."""
    print('\n...Creating directories.\n')
    if results_dir == './results/':
        results_dir = os.path.join(results_dir, modality,
                                   f'{latent_dim}d', optim, prior, str(spike), str(slab), str(pi), f'seed{rnd_seed:02d}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if plots_dir == './plots/':
        plots_dir = os.path.join(plots_dir, modality,
                                 f'{latent_dim}d', optim, prior, str(spike), str(slab), str(pi), f'seed{rnd_seed:02d}')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, 'model')
    return results_dir, plots_dir, model_dir


def run(
        task: str,
        modality: str,
        results_dir: str,
        plots_dir: str,
        triplets_dir: str,
        epochs: int,
        eta: float,
        batch_size: int,
        latent_dim: int,
        optim: str,
        prior: str,
        mc_samples: int,
        spike: float,
        slab: float,
        pi: float,
        steps: int,
        device: torch.device,
        rnd_seed: int,
        verbose: bool = True,
) -> None:
    """Perform VICE training."""
    # load triplets into memory
    train_triplets, test_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir)
    N = train_triplets.shape[0]
    n_items = utils.get_nitems(train_triplets)
    train_batches, val_batches = utils.load_batches(
        train_triplets=train_triplets,
        test_triplets=test_triplets,
        n_items=n_items,
        batch_size=batch_size,
        sampling_method='normal',
        rnd_seed=rnd_seed,
    )
    print(f'\nNumber of train batches: {len(train_batches)}\n')

    # initialize VICE model
    vice = VICE(prior=prior, in_size=n_items, out_size=latent_dim, init_weights=True)
    vice.to(device)
 
    results_dir, plots_dir, model_dir = create_dirs(
        results_dir=results_dir,
        plots_dir=plots_dir,
        modality=modality,
        latent_dim=latent_dim,
        optim=optim,
        prior=prior,
        spike=spike,
        slab=slab,
        pi=pi,
        rnd_seed=rnd_seed,
    )
    # initialize trainer
    trainer = Trainer(
        model=vice,
        task=task,
        N=N,
        n_items=n_items,
        latent_dim=latent_dim,
        optim=optim,
        eta=eta,
        batch_size=batch_size,
        epochs=epochs,
        mc_samples=mc_samples,
        prior=prior,
        spike=spike,
        slab=slab,
        pi=pi,
        steps=steps,
        model_dir=model_dir,
        results_dir=results_dir,
        device=device,
        verbose=verbose,
    )
    # start training
    trainer.train(train_batches=train_batches, val_batches=val_batches)
    train_accs = trainer.train_accs
    val_accs = trainer.val_accs
    loglikelihoods = trainer.loglikelihoods
    complexity_losses = trainer.complexity_losses
    # get model parameters
    params = trainer.parameters

    visualization.plot_single_performance(
        plots_dir=plots_dir, val_accs=val_accs, train_accs=train_accs[::steps])
    visualization.plot_complexities_and_loglikelihoods(
        plots_dir=plots_dir, loglikelihoods=loglikelihoods, complexity_losses=complexity_losses)

    # compress model params and store as binary files
    with open(os.path.join(results_dir, 'parameters.npz'), 'wb') as f:
        np.savez_compressed(f, W_loc=params[0], W_scale=params[1])


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    if args.device != 'cpu':
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
        device = torch.device(args.device)

    run(
        task=args.task,
        modality=args.modality,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        triplets_dir=args.triplets_dir,
        epochs=args.epochs,
        eta=args.eta,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        optim=args.optim,
        prior=args.prior,
        mc_samples=args.mc_samples,
        spike=args.spike,
        slab=args.slab,
        pi=args.pi,
        steps=args.steps,
        device=device,
        rnd_seed=args.rnd_seed,
        verbose=args.verbose,
    )
