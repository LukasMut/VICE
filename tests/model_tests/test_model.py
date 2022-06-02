#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import json
import shutil
import torch
import unittest
import utils
import model
import copy

import numpy as np
import tests.helper as helper
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

batch_size = 128
test_dir = "./test"
model_dir = os.path.join(test_dir, "model")
hypers = helper.get_hypers()
triplets = helper.create_triplets()
M = utils.get_nobjects(triplets)
subsample = triplets[
    np.random.choice(triplets.shape[0], size=batch_size, replace=False)
]
prior = "gaussian"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VICETestCase(unittest.TestCase):
    def get_model(self, hypers: dict):
        vice = getattr(model, "VICE")(
            n_train=hypers["N"],
            n_objects=hypers["M"],
            init_dim=hypers["P"],
            optim=hypers["optim"],
            eta=hypers["eta"],
            batch_size=hypers["batch_size"],
            epochs=hypers["epochs"],
            burnin=hypers["burnin"],
            mc_samples=hypers["mc_samples"],
            prior=hypers["prior"],
            spike=hypers["spike"],
            slab=hypers["slab"],
            pi=hypers["pi"],
            k=hypers["k"],
            ws=hypers["ws"],
            steps=hypers["steps"],
            model_dir=model_dir,
            results_dir=test_dir,
            device=device,
            verbose=True,
            init_weights=True,
        )
        vice.to(device)
        return vice

    def test_attributes(self):
        vice = self.get_model(hypers)
        self.assertTrue(issubclass(getattr(model, "Trainer"), nn.Module))
        self.assertTrue(hasattr(vice, "cuda" if torch.cuda.is_available() else "cpu"))

        self.assertTrue(hasattr(vice, "mu"))
        self.assertTrue(hasattr(vice, "sigma"))
        self.assertTrue(vice.prior, "gaussian")

        self.assertEqual(len(torch.unique(vice.sigma.logsigma.weight.data)), 1)

        self.assertTrue(vice.sigma.logsigma.weight.data.sum() < 0.0)
        self.assertTrue(vice.sigma.logsigma.weight.data.exp().min() >= 0.0)

        self.assertEqual(vice.mu.mu.in_features, vice.sigma.logsigma.in_features, M)

        self.assertEqual(
            vice.mu.mu.out_features, vice.sigma.logsigma.out_features, int(M / 2)
        )

    def test_output(self):
        vice = self.get_model(hypers)
        batches = utils.load_batches(
            train_triplets=None,
            test_triplets=subsample,
            n_objects=M,
            batch_size=batch_size,
            inference=True,
        )
        for batch in batches:
            self.assertEqual(batch.shape, (int(batch_size * 3), M))
            out = vice(batch)
            self.assertEqual(len(out), 4)
            z, W_mu, W_sigma, W_sample = out
            self.assertTrue(z.min() >= 0.0)
            self.assertTrue(W_sigma.min() >= 0.0)
            self.assertEqual(W_mu.shape, W_sigma.shape, W_sample.shape)
            self.assertEqual(z.shape, (int(batch_size * 3), int(M / 2)))

    def test_properties(self) -> None:
        vice = self.get_model(hypers)
        self.assertEqual(vice.n_train, hypers["N"])
        self.assertEqual(vice.n_objects, hypers["M"])
        self.assertEqual(vice.init_dim, hypers["P"])
        self.assertEqual(vice.optim, hypers["optim"])
        self.assertEqual(vice.eta, hypers["eta"])
        self.assertEqual(vice.batch_size, hypers["batch_size"])
        self.assertEqual(vice.epochs, hypers["epochs"])
        self.assertEqual(vice.mc_samples, hypers["mc_samples"])
        self.assertEqual(vice.prior, hypers["prior"])
        self.assertEqual(vice.spike, hypers["spike"])
        self.assertEqual(vice.slab, hypers["slab"])
        self.assertEqual(vice.pi, hypers["pi"])
        self.assertEqual(vice.steps, hypers["steps"])
        self.assertEqual(vice.device, device)
        self.assertTrue(vice.verbose)

        vice.initialize_priors_()
        self.assertTrue(hasattr(vice, "loc"))
        self.assertTrue(hasattr(vice, "scale_spike"))
        self.assertTrue(hasattr(vice, "scale_slab"))

        np.testing.assert_allclose(
            vice.loc, torch.zeros(hypers["M"], hypers["P"]).to(device)
        )
        np.testing.assert_allclose(
            vice.scale_spike,
            torch.ones(hypers["M"], hypers["P"]).mul(hypers["spike"]).to(device),
        )
        np.testing.assert_allclose(
            vice.scale_slab,
            torch.ones(hypers["M"], hypers["P"]).mul(hypers["slab"]).to(device),
        )

        params = vice.detached_params
        np.testing.assert_allclose(
            params["loc"], vice.mu.mu.weight.data.T.cpu().numpy()
        )
        np.testing.assert_allclose(
            params["scale"], vice.sigma.logsigma.weight.data.T.exp().cpu().numpy()
        )

    def test_optimization(self) -> None:
        vice = self.get_model(hypers)
        # get detached model parameters at initilization / start of training
        params_init = copy.deepcopy(vice.detached_params)

        train_triplets, test_triplets = helper.create_train_test_split(triplets)
        train_batches, val_batches = utils.load_batches(
            train_triplets=train_triplets,
            test_triplets=test_triplets,
            n_objects=hypers["M"],
            batch_size=hypers["batch_size"],
            inference=False,
        )

        vice.fit(train_batches=train_batches, val_batches=val_batches)

        # get model paramters after optimization / end of training
        params_opt = copy.deepcopy(vice.detached_params)

        # check whether model parameters have changed during optimization
        self.assertTrue(
            self.assert_difference(params_init["loc"], params_opt["loc"]))
        self.assertTrue(
            self.assert_difference(params_init["scale"], params_opt["scale"])
        )

        val_loss, val_acc = vice.evaluate(val_batches)

        self.assertTrue(type(val_loss) == float)
        self.assertTrue(type(val_acc) == float)
        self.assertTrue(val_loss < np.log(3))
        self.assertTrue(val_acc > 1 / 3)

        vice.fit(train_batches=train_batches, val_batches=val_batches)
        # examine whether model parameters haven't changed after loading from checkpoint
        np.testing.assert_allclose(params_opt["loc"], vice.detached_params["loc"])
        np.testing.assert_allclose(params_opt["scale"], vice.detached_params["scale"])

        hypers["epochs"] *= 2
        vice = self.get_model(hypers)
        vice.fit(train_batches=train_batches, val_batches=val_batches)
        self.assertTrue(
            self.assert_difference(params_opt["loc"], vice.detached_params["loc"])
        )
        self.assertTrue(
            self.assert_difference(params_opt["scale"], vice.detached_params["scale"])
        )

        # get pruned model params after optimization
        pruned_params = vice.pruned_params
        pruned_loc = pruned_params["pruned_loc"]
        pruned_scale = pruned_params["pruned_scale"]

        self.assertTrue(isinstance(pruned_params, dict))
        self.assertTrue(isinstance(pruned_loc, np.ndarray))
        self.assertTrue(isinstance(pruned_scale, np.ndarray))

        self.assertTrue(pruned_loc.shape[1] <= params_opt['loc'].shape[1])
        self.assertTrue(pruned_scale.shape[1] <= params_opt['scale'].shape[1])

    @staticmethod
    def assert_difference(A: np.ndarray, B: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(A, B)
            return False
        except AssertionError:
            return True

    def test_results(self) -> None:
        results = []
        regex = r"(?=.*\d)(?=.*json$)"
        for root, _, files in os.walk(test_dir):
            for f in files:
                if re.compile(regex).search(f):
                    with open(os.path.join(root, f), "r") as rf:
                        r = json.load(rf)
                    self.assertTrue(isinstance(r, dict))
                    results.append(r)
        self.assertTrue(len(results) <= int(hypers["epochs"] / hypers["steps"]))
        shutil.rmtree(test_dir)
