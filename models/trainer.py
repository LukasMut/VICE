#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import torch
import utils
import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from torch.optim import SGD, Adam, AdamW
from typing import Any, Dict, Iterator, List, Tuple


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
        burnin: int,
        mc_samples: int,
        prior: str,
        spike: float,
        slab: float,
        pi: float,
        k: int,
        ws: int,
        steps: int,
        model_dir: str,
        results_dir: str,
        device: torch.device,
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
        self.burnin = burnin
        self.mc_samples = mc_samples  # number of weight samples M
        self.prior = prior  # Gaussian or Laplace prior
        self.spike = spike
        self.slab = slab
        self.pi = pi
        self.k = k
        self.ws = ws
        self.steps = steps
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.device = device
        self.verbose = verbose

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
                    self.start = checkpoint['epoch']
                    self.loss = checkpoint['loss']
                    self.train_accs = checkpoint['train_accs']
                    self.val_accs = checkpoint['val_accs']
                    self.train_losses = checkpoint['train_losses']
                    self.val_losses = checkpoint['val_losses']
                    self.loglikelihoods = checkpoint['loglikelihoods']
                    self.complexity_losses = checkpoint['complexity_costs']
                    self.latent_causes = checkpoint['latent_causes']
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
                    self.latent_causes = []
            else:
                self.start = 0
                self.train_accs, self.val_accs = [], []
                self.train_losses, self.val_losses = [], []
                self.loglikelihoods, self.complexity_losses = [], []
                self.latent_causes = []
        else:
            os.makedirs(self.model_dir)
            self.start = 0
            self.train_accs, self.val_accs = [], []
            self.train_losses, self.val_losses = [], []
            self.loglikelihoods, self.complexity_losses = [], []
            self.latent_causes = []

    def initialize_priors_(self) -> None:
        self.loc = torch.zeros(self.n_items, self.latent_dim).to(self.device)
        self.scale_spike = torch.ones(self.n_items, self.latent_dim).mul(
            self.spike).to(self.device)
        self.scale_slab = torch.ones(self.n_items, self.latent_dim).mul(
            self.slab).to(self.device)

    def initialize_optim_(self) -> None:
        if self.optim == 'adam':
            self.optim = Adam(self.parameters(), eps=1e-08,
                              lr=self.eta)
        elif self.optim == 'adamw':
            self.optim = AdamW(self.parameters(),
                               eps=1e-08, lr=self.eta)
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
            return (pos_sim, neg_sim, neg_sim_2)
        return (pos_sim, neg_sim)

    @staticmethod
    def break_ties(probas: np.ndarray) -> np.ndarray:
        return np.array([-1 if len(np.unique(pmf)) != len(pmf) else np.argmax(pmf) for pmf in probas])

    def accuracy_(self, probas: np.ndarray, batching: bool = True) -> float:
        choices = self.break_ties(probas)
        argmax = np.where(choices == 0, 1, 0)
        acc = argmax.mean() if batching else argmax.tolist()
        return acc

    @staticmethod
    def sumexp(sims: Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.sum(torch.exp(torch.stack(sims)), dim=0)

    def softmax(self, sims: Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.exp(sims[0]) / self.sumexp(sims)

    def logsumexp(self, sims: Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.log(self.sumexp(sims))

    def log_softmax(self, sims: Tuple[torch.Tensor]) -> torch.Tensor:
        return sims[0] - self.logsumexp(sims)

    def cross_entropy_loss(self, sims: Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.mean(-self.log_softmax(sims))

    def choice_accuracy(self, similarities: float) -> float:
        probas = F.softmax(torch.stack(similarities, dim=-1),
                           dim=1).detach().cpu().numpy()
        choice_acc = self.accuracy_(probas)
        return choice_acc

    def pruning(self, alpha: float = .05,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        loc = self.detached_params['loc']
        scale = self.detached_params['scale']
        p_vals = utils.compute_pvals(loc, scale)
        rejections = utils.fdr_corrections(p_vals, alpha)
        importance = utils.get_importance(rejections).ravel()
        signal = np.where(importance > self.k)[0]
        pruned_loc = loc[:, signal]
        pruned_scale = scale[:, signal]
        return signal, pruned_loc, pruned_scale

    @staticmethod
    def convergence(latent_causes: List[int], ws: int) -> bool:
        """Evaluate convergence of latent causes."""
        causes_over_time = set(latent_causes[-ws:])
        divergence = len(causes_over_time)
        if (divergence == 1 and causes_over_time.pop() != 0):
            return True
        return False

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
                soft_choices = self.softmax(similarities)
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
        """Evaluate model on validation set."""
        self.eval()
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
        """Perform inference on a held-out test set."""
        probas = torch.zeros(int(len(test_batches) * self.batch_size), 3)
        triplet_choices = []
        model_choices = defaultdict(list)
        self.eval()
        batch_accs = torch.zeros(len(test_batches))
        batch_centropies = torch.zeros(len(test_batches))
        for j, batch in enumerate(test_batches):
            batch = batch.to(self.device)
            test_acc, test_loss, batch_probas, batch_choices = self.mc_sampling(
                batch)
            triplet_choices.extend(batch_choices)
            try:
                probas[j * self.batch_size:(j + 1)
                    * self.batch_size] += batch_probas
            except RuntimeError:
                probas[j * self.batch_size:(j
                    * self.batch_size) + batch_probas.shape[0]] += batch_probas
            batch_accs[j] += test_acc
            batch_centropies[j] += test_loss
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
        """Step over the full training data in mini-batches of size B."""
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

            signal, _, _ = self.pruning()
            n_latents = signal.shape[0]
            self.latent_causes.append(n_latents)

            if self.verbose:
                print(
                    "\n======================================================================================")
                print(
                    f'====== Epoch: {epoch+1:02d}, Train acc: {avg_train_acc:.3f}, Train loss: {avg_train_loss:.3f}, Latent causes: {n_latents:02d} ======')
                print(
                    "======================================================================================\n")

            if (epoch + 1) % self.steps == 0:
                avg_val_loss, avg_val_acc = self.evaluate(val_batches)
                self.val_losses.append(avg_val_loss)
                self.val_accs.append(avg_val_acc)
                self.save_checkpoint(epoch)
                self.save_results(epoch)

            if epoch > self.burnin:
                # evaluate model convergence
                if self.convergence(self.latent_causes, self.ws):
                    self.save_checkpoint(epoch)
                    self.save_results(epoch)
                    print('\n...Stopping VICE optimzation.')
                    print(
                        f'Latent dimensionality converged after {epoch+1:02d} epochs.\n')
                    break

        self.save_final_latents()

    def save_checkpoint(self, epoch: int) -> None:
        # PyTorch convention is to save checkpoints as .tar files
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': copy.deepcopy(self.state_dict()),
            'optim_state_dict': copy.deepcopy(self.optim.state_dict()),
            'loss': self.loss,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'loglikelihoods': self.loglikelihoods,
            'complexity_costs': self.complexity_losses,
            'latent_causes': self.latent_causes,
        }
        torch.save(checkpoint, os.path.join(
            self.model_dir, f'model_epoch{epoch+1:04d}.tar'))

    def save_results(self, epoch: int) -> None:
        results = {'epoch': epoch + 1,
                   'train_acc': self.train_accs[-1], 'val_acc': self.val_accs[-1], 'val_loss': self.val_losses[-1]}
        with open(os.path.join(self.results_dir, f'results_{epoch+1:04d}.json'), 'w') as rf:
            json.dump(results, rf)

    def save_final_latents(self):
        _, pruned_loc, pruned_scale = self.pruning()
        with open(os.path.join(self.results_dir, 'pruned_params.npz'), 'wb') as f:
            np.savez_compressed(f, pruned_loc=pruned_loc,
                                pruned_scale=pruned_scale)
