#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import torch
import utils
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from typing import Any, Iterator


os.environ['PYTHONIOENCODING'] = 'UTF-8'

class Trainer(object):

    def __init__(
                self,
                model: nn.Module,
                task: str,
                N: int,
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
                temperature: float,
                steps: int,
                model_dir: str,
                results_dir: str,
                device: torch.device,
                verbose: bool=False,
    ):

    self.model = model
    self.task = task
    self.N = N # number of trials/triplets in dataset
    self.n_items = n_items # number of unique items/objects
    self.latent_dim = latent_dim
    self.optim = optim
    self.eta = eta
    self.batch_size = batch_size
    self.epochs = epochs
    self.mc_samples = mc_samples
    self.prior
    self.spike = spike
    self.slab = slab
    self.pi = pi
    self.temperature = temperature
    self.model_dir = model_dir
    self.results_dir = results_dir
    self.device = device
    self.verbose = verbose

    def initialize_priors_(self):
        self.mu = torch.zeros(self.n_items, self.laten_dim).to(self.device)
        self.sigma_spike = torch.ones(self.n_items, self.latent_dim).mul(self.spike).to(self.device)
        self.sigma_slab = torch.ones(self.n_items, self.latent_dim).mul(self.slab).to(self.device)


    def pdf(self, W_sample: torch.Tensor, W_mu: torch.Tensor, W_sigma: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((W_sample - W_mu) ** 2) / (2 * W_sigma.pow(2))) / W_sigma * math.sqrt(2 * math.pi)


    def spike_and_slab(self, W_sample: torch.Tensor) -> torch.Tensor:
        spike = self.pi * self.pdf(W_sample, self.mu, self.sigma_spike)
        slab = (1 - self.pi) * self.pdf(W_sample, self.mu, self.sigma_slab)
        return spike + slab


    def evaluate(self, val_batches: Iterator) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            batch_losses_val = torch.zeros(len(val_batches))
            batch_accs_val = torch.zeros(len(val_batches))
            for j, batch in enumerate(val_batches):
                batch = batch.to(device)
                val_acc, val_loss, _ = self.mc_sampling(batch)
                batch_losses_val[j] += val_loss.item()
                batch_accs_val[j] += val_acc
        avg_val_loss = torch.mean(batch_losses_val).item()
        avg_val_acc = torch.mean(batch_accs_val).item()
        return avg_val_loss, avg_val_acc


    def mc_sampling(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_choices = 3 if self.task == 'odd_one_out' else 2
        sampled_probas = torch.zeros(self.mc_samples, batch.shape[0] // n_choices, n_choices).to(self.device)
        sampled_choices = torch.zeros(self.mc_samples, batch.shape[0] // n_choices).to(self.device)
        for k in range(self.mc_samples):
            logits, _, _, _ = self.model(batch)
            anchor, positive, negative = torch.unbind(torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1)
            similarities = utils.compute_similarities(anchor, positive, negative, self.task)
            soft_choices = utils.softmax(similarities, self.temperature)
            probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
            sampled_probas[k] += probas
            sampled_choices[k] +=  soft_choices
        probas = sampled_probas.mean(dim=0)
        val_acc = utils.accuracy_(probas.cpu().numpy())
        soft_choices = sampled_choices.mean(dim=0)
        val_loss = torch.mean(-torch.log(soft_choices))
        return val_acc, val_loss, probas


    def stepping(self, train_batches):
        batch_lilkelihoods = torch.zeros(len(train_batches))
        batch_closses = torch.zeros(len(train_batches))
        batch_losses = torch.zeros(len(train_batches))
        batch_accs = torch.zeros(len(train_batches))
        for i, batch in enumerate(train_batches):
            self.optim.zero_grad()
            batch = batch.to(self.device)
            logits, W_mu, W_sigma, W_sampled = self.model(batch)
            anchor, positive, negativ = torch.unbind(
                torch.reshape(logits, (-1, 3, self.latent_dim)), dim=1)
            c_entropy = utils.trinomial_loss(
                anchor, positive, negative, task, self.temperature)
            log_q = self.pdf(W_sampled, W_mu, W_sigma).log()
            log_p = self.spike_and_slab(W_sampled).log()
            complexity_loss = (1 / self.N) * (log_q.sum() - log_p.sum()))
            loss = c_entropy + complexity_loss
            loss.backward()
            self.optim.step()
            batch_losses[i] += loss.item()
            batch_llikelihoods[i] += c_entropy.item()
            batch_closses[i] += complexity_loss.item()
            batch_accs[i] += utils.choice_accuracy(
                anchor, positive, negative, task
            )
        return batch_llikelihoods, batch_closses, batch_losses, batch_accs

    def train(self, train_batches: Iterator, val_batches: Iterator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # initialize the spike and slab priors
        self.intitialize_priors()
        # start training
        for epoch in range(self.start, self.epochs):
            self.model.train()
            # take a step over the entire training data (i.e., iterate over every mini-batch)
            batch_llikelihoods, batch_closses, batch_losses, batch_accs = self.stepping(
                train_batches)

            avg_llikelihood = torch.mean(batch_llikelihoods).item()
            avg_closs = torch.mean(batch_closses).item()
            avg_train_loss = torch.mean(batch_losses).item()
            avg_train_acc = torch.mean(batch_accs).item()

            loglikelihoods.append(avg_llikelihood)
            complexity_losses.append(avg_closs)
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)

            if self.verbose:
                print("\n===============================================================================================")
                print(
                    f'====== Epoch: {epoch+1}, Train acc: {avg_train_acc:.3f}, Train loss: {avg_train_loss:.3f} ======')
                print("=================================================================================================\n")

            if (epoch + 1) % self.steps == 0:
                avg_val_loss, avg_val_acc = self.evaluate(val_batches)
                val_losses.append(avg_val_loss)
                val_accs.append(avg_val_acc)

            # save model and optim parameters for inference or to resume training
            # PyTorch convention is to save checkpoints as .tar files
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'loss': loss,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'loglikelihoods': loglikelihoods,
                'complexity_costs': complexity_losses,
            }, os.path.join(self.model_dir, f'model_epoch{epoch+1:04d}.tar'))

            results = {'epoch': len(
                train_accs), 'train_acc': train_accs[-1], 'val_acc': val_accs[-1], 'val_loss': val_losses[-1]}
            
            with open(os.path.join(self.results_dir, f'results_{epoch+1:04d}.json'), 'w') as rf:
                json.dump(results, rf)
        
        return val_accs, train_accs, train_losses, val_losses, loglikelihoods, complexity_losses