#!/usr/bin/env python3
# -*- coding: utf-8 -*


import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Dict, List

import numpy as np

os.environ["PYTHONIOENCODING"] = "UTF-8"
os.environ[
    "OMP_NUM_THREADS"
] = "1"  # number of cores used per Python process (set to 2 if HT is enabled, else keep 1)


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--in_path", type=str, help="path/to/models/and/pruning/results")
    aa("--percentages", type=int, nargs="+", 
        choices=[5, 10, 20, 50, 100],
        help="percentage of full dataset used within a subsample")
    args = parser.parse_args()
    return args


def get_split_results(PATH: str, fractions: List[int]) -> Dict[int, List[str]]:
    trees = defaultdict(list)
    for p in fractions:
        split_path = os.path.join(PATH, f"{p:02d}")
        for d in os.scandir(split_path):
            # check whether path refers to random seed
            if d.is_dir() and d.name[-2:].isdigit():
                roots = get_results(os.path.join(split_path, d.name))
                trees[p].extend(roots)
    return trees


def get_results(PATH: str) -> Dict[tuple, dict]:
    results = defaultdict(list)
    trees = defaultdict(list)
    for root, _, files in os.walk(PATH):
        for f in files:
            if f.endswith("2000.json"):
                hypers = tuple(map(float, root.split("/")[-4:-1]))
                with open(os.path.join(root, f), "r") as r:
                    val_centropy = json.load(r)["val_loss"]
                    if np.isnan(val_centropy):
                        print(f"Found NaN in cross-entropy loss for: {root}")
                        val_centropy = np.inf
                results[hypers].append(val_centropy)
                trees[hypers].append(root)
    avg_centropies = aggregate_centropies(results)
    if sum(np.isinf(list(avg_centropies.values()))) == len(results):
        raise Exception(
            "\nFound NaN values in cross-entropy error for every model. Change hyperparameter grid.\n"
        )
    best_comb = get_best_comb(avg_centropies)
    roots = trees[best_comb]
    del trees[best_comb]
    del_paths(trees)
    print(f"\nBest hyperparameter combination: {best_comb}")
    print(f"Average cross-entropy error: {avg_centropies[best_comb]}\n")
    return roots


def aggregate_centropies(results: Dict[tuple, list]) -> Dict[tuple, float]:
    return {hypers: np.mean(centropies) for hypers, centropies in results.items()}


def get_best_comb(avg_centropies: Dict[tuple, float]) -> tuple:
    return min(avg_centropies.items(), key=lambda kv: kv[1])[0]


def del_paths(trees: Dict[tuple, list]) -> None:
    for roots in trees.values():
        for root in roots:
            shutil.rmtree(root)


if __name__ == "__main__":
    args = parseargs()
    trees = get_split_results(args.in_path, args.percentages)

    for p, roots in trees.items():
        with open(os.path.join(args.in_path, f"{p:02d}", "model_paths.txt"), "w") as f:
            count = 0
            for root in roots:
                model_files = "/".join((root, "models"))
                f.write(model_files)
                f.write("\n")
                count += 1
            # assert count == int((100 // p) * 20)
