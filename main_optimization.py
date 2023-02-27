#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import optimization
import utils
import visualization
from data import TripletData

Array = np.ndarray
Tensor = torch.Tensor


os.environ["PYTHONIOENCODING"] = "UTF-8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--task",
        type=str,
        default="odd-one-out",
        choices=["odd-one-out", "target-matching"],
        help="whether to perform an odd-one-out (3AFC) or target-matching (2AFC) triplet task",
    )
    aa("--triplets_dir", type=str, help="directory from where to load triplets")
    aa(
        "--results_dir",
        type=str,
        default="./results/",
        help="optional specification of results directory (if not provided will resort to ./results/init_dim/optim/mixture/seed/spike/slab/pi)",
    )
    aa(
        "--plots_dir",
        type=str,
        default="./plots/",
        help="optional specification of directory for plots (if not provided will resort to ./plots/init_dim/optim/mixture/seed/spike/slab/pi)",
    )
    aa(
        "--epochs",
        metavar="T",
        type=int,
        default=2000,
        help="maximum number of epochs to run VICE optimization",
    )
    aa(
        "--burnin",
        type=int,
        default=500,
        help="minimum number of epochs to run VICE optimization",
    )
    aa("--eta", type=float, default=0.001, help="learning rate to be used in optimizer")
    aa(
        "--init_dim",
        metavar="D",
        type=int,
        default=100,
        help="initial dimensionality of the latent space",
    )
    aa(
        "--batch_size",
        metavar="B",
        type=int,
        default=128,
        help="number of triplets sampled during each step (i.e., mini-batch size)",
    )
    aa(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="optimizer to train VICE",
    )
    aa(
        "--mixture",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplace"],
        help="whether to use a Gaussian or Laplacian mixture for the spike-and-slab prior",
    )
    aa(
        "--mc_samples",
        type=int,
        default=10,
        help="number of weight samples to use for MC sampling",
    )
    aa("--spike", type=float, default=0.25, help="sigma for spike distribution")
    aa(
        "--slab",
        type=float,
        default=1.0,
        help="sigma for slab distribution (should be smaller than spike)",
    )
    aa(
        "--pi",
        type=float,
        default=0.5,
        help="scalar value that determines the relative weight of the spike and slab distributions respectively",
    )
    aa(
        "--k",
        type=int,
        default=5,
        choices=[5, 10],
        help="minimum number of items that have non-zero weight for a latent dimension (according to importance scores)",
    )
    aa(
        "--ws",
        type=int,
        default=500,
        help="determines for how many epochs the number of latent dimensions (after pruning) is not allowed to vary",
    )
    aa(
        "--steps",
        type=int,
        default=50,
        help="perform validation and save model parameters every <steps> epochs",
    )
    aa(
        "--device",
        type=str,
        default="cpu",
        help="whether training should be performed on CPU or GPU (i.e., CUDA).",
    )
    aa(
        "--num_threads",
        type=int,
        default=4,
        help="number of threads used for intraop parallelism on CPU; use only if device is CPU",
    )
    aa(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers used for loading data",
    )
    aa(
        "--rnd_seed",
        type=int,
        default=42,
        help="random seed for reproducibility of results",
    )
    aa(
        "--verbose",
        action="store_true",
        help="whether to display print statements about model performance during training",
    )
    args = parser.parse_args()
    return args


def create_dirs(
    results_dir: str,
    plots_dir: str,
    init_dim: int,
    optim: str,
    mixture: str,
    spike: float,
    slab: float,
    pi: float,
    rnd_seed: int,
) -> Tuple[str, str, str]:
    """Create directories for results, plots, and optimization parameters."""
    print("\n...Creating directories.\n")
    results_dir = os.path.join(
        results_dir,
        f"{init_dim}d",
        optim,
        mixture,
        str(spike),
        str(slab),
        str(pi),
        f"seed{rnd_seed:02d}",
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(
        plots_dir,
        f"{init_dim}d",
        optim,
        mixture,
        str(spike),
        str(slab),
        str(pi),
        f"seed{rnd_seed:02d}",
    )
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, "model")
    return results_dir, plots_dir, model_dir


def get_nobjects(train_triplets: Tensor) -> int:
    """Get number of unique items in the data."""
    n_objects = torch.max(train_triplets).item()
    if torch.min(train_triplets).item() == 0:
        n_objects += 1
    return n_objects


def run(
    results_dir: str,
    plots_dir: str,
    triplets_dir: str,
    task: str,
    epochs: int,
    burnin: int,
    eta: float,
    batch_size: int,
    init_dim: int,
    optim: str,
    mixture: str,
    mc_samples: int,
    spike: float,
    slab: float,
    pi: float,
    k: int,
    ws: int,
    steps: int,
    num_workers: int,
    device: torch.device,
    rnd_seed: int,
    verbose: bool = True,
) -> None:
    """Perform VICE training."""
    # load triplets into memory
    train_triplets, val_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir
    )
    N = train_triplets.shape[0]
    n_objects = get_nobjects(train_triplets)
    train_triplets = TripletData(
        triplets=train_triplets,
        n_objects=n_objects,
    )
    val_triplets = TripletData(
        triplets=val_triplets,
        n_objects=n_objects,
    )
    train_batches = DataLoader(
        dataset=train_triplets,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    val_batches = DataLoader(
        dataset=val_triplets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )
    print(f"\nNumber of train batches: {len(train_batches)}\n")
    results_dir, plots_dir, model_dir = create_dirs(
        results_dir=results_dir,
        plots_dir=plots_dir,
        init_dim=init_dim,
        optim=optim,
        mixture=mixture,
        spike=spike,
        slab=slab,
        pi=pi,
        rnd_seed=rnd_seed,
    )
    # initialize VICE model
    vice = optimization.VICE(
        task=task,
        n_train=N,
        n_objects=n_objects,
        init_dim=init_dim,
        optim=optim,
        eta=eta,
        batch_size=batch_size,
        epochs=epochs,
        burnin=burnin,
        mc_samples=mc_samples,
        mixture=mixture,
        spike=spike,
        slab=slab,
        pi=pi,
        k=k,
        ws=ws,
        steps=steps,
        model_dir=model_dir,
        results_dir=results_dir,
        device=device,
        verbose=verbose,
        init_weights=True,
    )
    # move model to current device
    vice.to(device)
    # start VICE training
    vice.fit(train_batches=train_batches, val_batches=val_batches)
    # get performance scores
    train_accs = vice.train_accs
    val_accs = vice.val_accs
    loglikelihoods = vice.loglikelihoods
    complexity_losses = vice.complexity_losses
    latent_dimensions = vice.latent_dimensions
    # get (detached) VICE parameters
    params = vice.detached_params

    visualization.plot_single_performance(
        plots_dir=plots_dir, val_accs=val_accs, train_accs=train_accs, steps=steps
    )
    visualization.plot_complexities_and_loglikelihoods(
        plots_dir=plots_dir,
        loglikelihoods=loglikelihoods,
        complexity_losses=complexity_losses,
    )
    visualization.plot_latent_dimensions(
        plots_dir=plots_dir, latent_dimensions=latent_dimensions
    )

    # compress model params and store as binary files
    with open(os.path.join(results_dir, "parameters.npz"), "wb") as f:
        np.savez_compressed(f, loc=params["loc"], scale=params["scale"])


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    if re.search(r"cuda", args.device):
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
        device = torch.device(f"cuda:{current_device}")
        print(f"\nPyTorch CUDA version: {torch.version.cuda}")
        print(f"Process is running on *cuda:{current_device}*\n")
    else:
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        torch.set_num_threads(args.num_threads)
        device = torch.device(args.device)

    run(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        triplets_dir=args.triplets_dir,
        task=args.task,
        epochs=args.epochs,
        burnin=args.burnin,
        eta=args.eta,
        batch_size=args.batch_size,
        init_dim=args.init_dim,
        optim=args.optim,
        mixture=args.mixture,
        mc_samples=args.mc_samples,
        spike=args.spike,
        slab=args.slab,
        pi=args.pi,
        k=args.k,
        ws=args.ws,
        steps=args.steps,
        num_workers=args.num_workers,
        device=device,
        rnd_seed=args.rnd_seed,
        verbose=args.verbose,
    )
