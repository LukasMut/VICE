#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import utils
import visualization
import model

import numpy as np

from typing import Tuple

os.environ["PYTHONIOENCODING"] = "UTF-8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
    print("\n...Creating directories.\n")
    results_dir = os.path.join(
        results_dir,
        modality,
        f"{latent_dim}d",
        optim,
        prior,
        str(spike),
        str(slab),
        str(pi),
        f"seed{rnd_seed:02d}",
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(
        plots_dir,
        modality,
        f"{latent_dim}d",
        optim,
        prior,
        str(spike),
        str(slab),
        str(pi),
        f"seed{rnd_seed:02d}",
    )
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    model_dir = os.path.join(results_dir, "model")
    return results_dir, plots_dir, model_dir


def run(
    task: str,
    modality: str,
    results_dir: str,
    plots_dir: str,
    triplets_dir: str,
    epochs: int,
    burnin: int,
    eta: float,
    batch_size: int,
    latent_dim: int,
    optim: str,
    prior: str,
    mc_samples: int,
    spike: float,
    slab: float,
    pi: float,
    k: int,
    ws: int,
    steps: int,
    device: torch.device,
    rnd_seed: int,
    verbose: bool = True,
) -> None:
    """Perform VICE training."""
    # load triplets into memory
    train_triplets, test_triplets = utils.load_data(
        device=device, triplets_dir=triplets_dir
    )
    N = train_triplets.shape[0]
    n_items = utils.get_nitems(train_triplets)
    train_batches, val_batches = utils.load_batches(
        train_triplets=train_triplets,
        test_triplets=test_triplets,
        n_items=n_items,
        batch_size=batch_size,
    )
    print(f"\nNumber of train batches: {len(train_batches)}\n")
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
    # initialize VICE model
    vice = getattr(model, "VICE")(
        task=task,
        n_train=N,
        n_items=n_items,
        latent_dim=latent_dim,
        optim=optim,
        eta=eta,
        batch_size=batch_size,
        epochs=epochs,
        burnin=burnin,
        mc_samples=mc_samples,
        prior=prior,
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
        np.savez_compressed(f, W_loc=params["loc"], W_scale=params["scale"])
