#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import torch
import utils

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from torch.optim import SGD, Adam, AdamW
from typing import Any, Dict, Iterator, Tuple


os.environ['PYTHONIOENCODING'] = 'UTF-8'


class Trainer(nn.Module):

    def __init__(
        self,
        task: str,
        n_train: int,
        n_items: int,
        latent_dim: int,
        optim: Any,
        eta: str,
        batch_size: int,
        epochs: int,
        mc_samples: int,
        prior: str,
        spike: float,
        slab: float,
        pi: float,
        steps: int,
        model_dir: str,
        results_dir: str,
        device: torch.device,
        temperature=None,
        verbose: bool = False,
    ):
        super(Trainer, self).__init__()
        self.task = task
        self.n_train = n_train  # number of trials/triplets in dataset
        self.n_items = n_items  # number of unique items/objects
        self.latent_dim = latent_dim
        self.optim = optim
        self.eta = eta  # learning rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.mc_samples = mc_samples  # number of weight samples M
        self.prior = prior  # Gaussian or Laplace prior
        self.spike = spike
        self.slab = slab
        self.pi = pi
        self.steps = steps
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.device = device
        self.verbose = verbose

        if temperature is None:
            self.temperature = torch.tensor(1.)
        else:
            self.temperature = temperature

    def forward(self, *input: Any) -> None:
        raise NotImplementedError

    def load_checkpoint_(self) -> None:
        """Load model and optimizer params from previous checkpoint, if available."""
        if os.path.exists(self.model_dir):
            models = sorted([m.name for m in os.scandir(
                self.model_dir) if m.name.endswith('tar')])
            if len(models) > 0:
                try:
                    PATH = os.path.join(self.model_dir, models[-1])
                    checkpoint = torch.load(PATH, map_location=self.device)
                    self.load_state_dict(checkpoint['model_state_dict'])
                    self.optim.load_state_dict(checkpoint['optim_state_dict'])
                    self.start = checkpoint['epoch'] + 1
                    self.loss = checkpoint['loss']
                    self.train_accs = checkpoint['train_accs']
                    self.val_accs = checkpoint['val_accs']
                    self.train_losses = checkpoint['train_losses']
                    self.val_losses = checkpoint['val_losses']
                    self.loglikelihoods = checkpoint['loglikelihoods']
                    self.complexity_losses = checkpoint['complexity_costs']
                    print(
                        f'...Loaded model and optimizer params from previous run. Resuming training at epoch {self.start}.\n')
                except RuntimeError:
                    print(
                        '...Loading model and optimizer params failed. Check whether you are currently using a different set of model parameters.')
                    print('...Starting model training from scratch.\n')
                    self.start = 0
                    self.train_accs, self.val_accs = [], []
                    self.train_losses, self.val_losses = [], []
                    self.loglikelihoods, self.complexity_losses = [], []
            else:
                self.start = 0
                self.train_accs, self.val_accs = [], []
                self.train_losses, self.val_losses = [], []
                self.loglikelihoods, self.complexity_losses = [], []
        else:
            os.makedirs(self.model_dir)
            self.start = 0
            self.train_accs, self.val_accs = [], []
            self.train_losses, self.val_losses = [], []
            self.loglikelihoods, self.complexity_losses = [], []

    def initialize_priors_(self) -> None:
        self.loc = torch.zeros(self.n_items, self.latent_dim).to(self.device)
        self.scale_spike = torch.ones(self.n_items, self.latent_dim).mul(
            self.spike).to(self.device)
        self.scale_slab = torch.ones(self.n_items, self.latent_dim).mul(
            self.slab).to(self.device)

    def initialize_optim_(self) -> None:
        if self.optim == 'adam':
            self.optim = Adam(self.parameters(), betas=(
                0.9, 0.999), eps=1e-08, lr=self.eta)
        elif self.optim == 'adamw':
            self.optim = AdamW(self.parameters(), betas=(
                0.9, 0.999), eps=1e-08, lr=self.eta)
        else:
            self.optim = SGD(self.parameters(), lr=self.eta)

    @staticmethod
    def norm_pdf(W_sample: torch.Tensor, W_loc: torch.Tensor, W_scale: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((W_sample - W_loc) ** 2) / (2 * W_scale.pow(2))) / W_scale * math.sqrt(2 * math.pi)

    @staticmethod
    def laplace_pdf(W_sample: torch.Tensor, W_loc: torch.Tensor, W_scale: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(W_sample - W_loc).abs() / W_scale) / W_scale.mul(2.)

    def spike_and_slab(self, W_sample: torch.Tensor) -> torch.Tensor:
        pdf = self.norm_pdf if self.prior == 'gaussian' else self.laplace_pdf
        spike = self.pi * pdf(W_sample, self.loc, self.scale_spike)
        slab = (1 - self.pi) * pdf(W_sample, self.loc, self.scale_slab)
        return spike + slab

    @staticmethod
    def compute_similarities(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, task: str,
                             ) -> tuple:
        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)
        if task == 'odd_one_out':
            neg_sim_2 = torch.sum(positive * negative, dim=1)
            return pos_sim, neg_sim, neg_sim_2
        else:
            return pos_sim, neg_sim

    @staticmethod
    def softmax(sims: tuple, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(sims[0] / t) / torch.sum(torch.stack([torch.exp(sim / t) for sim in sims]), dim=0)

    @staticmethod
    def break_ties(probas: np.ndarray) -> np.ndarray:
        return np.array([-1 if len(np.unique(pmf)) != len(pmf) else np.argmax(pmf) for pmf in probas])

    def accuracy_(self, probas: np.ndarray, batching: bool = True) -> float:
        choices = self.break_ties(probas)
        argmax = np.where(choices == 0, 1, 0)
        acc = argmax.mean() if batching else argmax.tolist()
        return acc

    def cross_entropy_loss(self, similarities: float) -> torch.Tensor:
        return torch.mean(-torch.log(self.softmax(similarities, self.temperature)))

    def choice_accuracy(self, similarities: float) -> float:
        probas = F.softmax(torch.stack(similarities, dim=-1),
                           dim=1).detach().cpu().numpy()
        choice_acc = self.accuracy_(probas)
        return choice_acc

    def mc_sampling(self, batch: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform Monte Carlo sampling."""
        n_choices = 3 if self.task == 'odd_one_out' else 2
        sampled_probas = torch.zeros(
            self.mc_samples, batch.shape[0] // n_choices, n_choices).to(self.device)
        sampled_choices = torch.zeros(
            self.mc_samples, batch.shape[0] // n_choices).to(self.device)
        with torch.no_grad():
            for k in range(self.mc_samples):
                logits, _, _, _ = self.forward(batch)
                anchor, positive, negative = torch.unbind(
                    torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)
                similarities = self.compute_similarities(
                    anchor, positive, negative, self.task)
                soft_choices = self.softmax(similarities, self.temperature)
                probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
                sampled_probas[k] += probas
                sampled_choices[k] += soft_choices
        probas = sampled_probas.mean(dim=0)
        val_acc = self.accuracy_(probas.cpu().numpy())
        hard_choices = self.accuracy_(probas.cpu().numpy(), batching=False)
        soft_choices = sampled_choices.mean(dim=0)
        val_loss = torch.mean(-torch.log(soft_choices))
        return val_acc, val_loss, probas, hard_choices

    def evaluate(self, val_batches: Iterator) -> Tuple[float, float]:
        """Evaluate model."""
        self.eval()
        with torch.no_grad():
            batch_losses_val = torch.zeros(len(val_batches))
            batch_accs_val = torch.zeros(len(val_batches))
            for j, batch in enumerate(val_batches):
                batch = batch.to(self.device)
                val_acc, val_loss, _, _ = self.mc_sampling(batch)
                batch_losses_val[j] += val_loss.item()
                batch_accs_val[j] += val_acc
        avg_val_loss = torch.mean(batch_losses_val).item()
        avg_val_acc = torch.mean(batch_accs_val).item()
        return avg_val_loss, avg_val_acc

    def inference(self, test_batches: Iterator,
                  ) -> Tuple[float, float, np.ndarray, Dict[tuple, list]]:
        """Perform inference."""
        probas = torch.zeros(int(len(test_batches) * self.batch_size), 3)
        triplet_choices = []
        model_choices = defaultdict(list)
        self.eval()
        with torch.no_grad():
            batch_accs = torch.zeros(len(test_batches))
            batch_centropies = torch.zeros(len(test_batches))
            for j, batch in enumerate(test_batches):
                batch = batch.to(self.device)
                test_acc, test_loss, batch_probas, batch_choices = self.mc_sampling(
                    batch)
                triplet_choices.extend(batch_choices)
                probas[j * self.batch_size:(j + 1) * self.batch_size] += batch_probas
                batch_accs[j] += test_acc
                batch_centropies += test_loss
                human_choices = batch.nonzero(
                    as_tuple=True)[-1].view(self.batch_size, -1).cpu().numpy()
                model_choices = utils.collect_choices(
                    batch_probas, human_choices, model_choices)
        probas = probas.cpu().numpy()
        probas = probas[np.where(probas.sum(axis=1) != 0.)]
        model_pmfs = utils.compute_pmfs(model_choices, behavior=False)
        test_acc = batch_accs.mean().item()
        test_loss = batch_centropies.mean().item()
        return test_acc, test_loss, probas, model_pmfs, triplet_choices

    def stepping(self, train_batches: Iterator,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step over the full train set."""
        batch_llikelihoods = torch.zeros(len(train_batches))
        batch_closses = torch.zeros(len(train_batches))
        batch_losses = torch.zeros(len(train_batches))
        batch_accs = torch.zeros(len(train_batches))
        for i, batch in enumerate(train_batches):
            self.optim.zero_grad()
            batch = batch.to(self.device)
            logits, W_loc, W_scale, W_sampled = self.forward(batch)
            anchor, positive, negative = torch.unbind(
                torch.reshape(logits, (-1, 3, self.latent_dim)), dim=1)
            c_entropy = self.cross_entropy_loss(
                self.compute_similarities(anchor, positive, negative, self.task))

            if self.prior == 'gaussian':
                log_q = self.norm_pdf(W_sampled, W_loc, W_scale).log()
            else:
                log_q = self.laplace_pdf(W_sampled, W_loc, W_scale).log()

            log_p = self.spike_and_slab(W_sampled).log()
            complexity_loss = (1 / self.n_train) * (log_q.sum() - log_p.sum())
            self.loss = c_entropy + complexity_loss
            self.loss.backward()
            self.optim.step()
            batch_losses[i] += self.loss.item()
            batch_llikelihoods[i] += c_entropy.item()
            batch_closses[i] += complexity_loss.item()
            batch_accs[i] += self.choice_accuracy(
                self.compute_similarities(anchor, positive, negative, self.task))
        return batch_llikelihoods, batch_closses, batch_losses, batch_accs

    def fit(self, train_batches: Iterator, val_batches: Iterator) -> None:
        self.initialize_priors_()
        self.initialize_optim_()
        self.load_checkpoint_()
        for epoch in range(self.start, self.epochs):
            self.train()
            # take a step over the entire training data (i.e., iterate over every mini-batch)
            batch_llikelihoods, batch_closses, batch_losses, batch_accs = self.stepping(
                train_batches)

            avg_llikelihood = torch.mean(batch_llikelihoods).item()
            avg_closs = torch.mean(batch_closses).item()
            avg_train_loss = torch.mean(batch_losses).item()
            avg_train_acc = torch.mean(batch_accs).item()

            self.loglikelihoods.append(avg_llikelihood)
            self.complexity_losses.append(avg_closs)
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(avg_train_acc)

            if self.verbose:
                print(
                    "\n===============================================================================================")
                print(
                    f'====== Epoch: {epoch+1}, Train acc: {avg_train_acc:.3f}, Train loss: {avg_train_loss:.3f} ======')
                print(
                    "=================================================================================================\n")

            if (epoch + 1) % self.steps == 0:
                avg_val_loss, avg_val_acc = self.evaluate(val_batches)
                self.val_losses.append(avg_val_loss)
                self.val_accs.append(avg_val_acc)

                # save model and optim parameters for inference or to resume training at a later point
                # PyTorch convention is to save checkpoints as .tar files
                checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.state_dict(),
                            'optim_state_dict': self.optim.state_dict(),
                            'loss': self.loss,
                            'train_losses': self.train_losses,
                            'train_accs': self.train_accs,
                            'val_losses': self.val_losses,
                            'val_accs': self.val_accs,
                            'loglikelihoods': self.loglikelihoods,
                            'complexity_costs': self.complexity_losses,
                        }
                torch.save(checkpoint, os.path.join(self.model_dir, f'model_epoch{epoch+1:04d}.tar'))

                results = {'epoch': len(
                    self.train_accs), 'train_acc': self.train_accs[-1], 'val_acc': self.val_accs[-1], 'val_loss': self.val_losses[-1]}
                self.save_results(self.results_dir, epoch, results)

    @staticmethod
    def save_results(out_path: str, epoch: int, results: dict) -> None:
        with open(os.path.join(out_path, f'results_{epoch+1:04d}.json'), 'w') as rf:
            json.dump(results, rf)
