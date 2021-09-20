#!/usr/bin/env python3
# -*- coding: utf-8 -*


import argparse
import os
import pickle
import re
import shutil
import json
import numpy as np

from collections import defaultdict
from typing import List, Dict

os.environ['PYTHONIOENCODING']='UTF-8'
os.environ['OMP_NUM_THREADS']='1' #number of cores used per Python process (set to 2 if HT is enabled, else keep 1)


def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--in_path', type=str,
        help='path/to/models/and/pruning/results')
    aa('--percentages', type=int, nargs='+',
        choices=[5, 10, 20, 50, 100],
        help='percentage of full dataset used within a subsample')
    aa('--thresh', type=float,
        choices=[0.75, 0.8, 0.85, 0.9, 0.95],
        help='reproducibility threshold')
    aa('--seeds', type=int, nargs='+',
        help='list of random seeds used for training')
    args = parser.parse_args()
    return args


def get_split_results(
                        PATH: str,
                        percentages: List[int],
                        thresh: float,
) -> Tuple[Dict[str, dict], Dict[List[str]]]:
    split_results = defaultdict(lambda: defaultdict(dict))
    trees = defaultdict(list)
    for p in percentages:
        split_path = os.path.join(PATH, f'{p:02d}')
        for d in os.scandir(split_path):
            # check whether path refers to random seed
            if d.is_dir() and d.name[-2:].isdigit():
                results = get_results(os.path.join(split_path, d.name), thresh)
                best_comb = get_best_comb(results)
                del_paths(os.path.join(split_path, d.name), best_comb)
                tuning_cross_entropies = results[best_comb]['tuning_cross_entropies']
                pruning_cross_entropies = results[best_comb]['pruning_cross_entropies']
                robustness = results[best_comb]['robustness'] 
                root = results[best_comb]['root']
                split_results[p]['_'.join(d.name.split('_')[-2:])]['tuning_cross_entropies'] = list(tuning_cross_entropies)
                split_results[p]['_'.join(d.name.split('_')[-2:])]['pruning_cross_entropies'] = list(pruning_cross_entropies)
                split_results[p]['_'.join(d.name.split('_')[-2:])]['robustness'] = robustness
                split_results[p]['_'.join(d.name.split('_')[-2:])]['best_comb'] = best_comb
                trees[p].append(root)
    return split_results, trees


def get_results(PATH: str, thresh: float) -> Dict[tuple, dict]:
    results = defaultdict(dict)
    nans = []
    for root, _, files in os.walk(PATH):
        for f in files:
            path_list = root.split('/')
            if re.search(r'(?=^robust)(?=.*txt$)', f):
                m = re.compile(r'(?<=\.)\d+').search(root)
                start, end = m.span()
                corr = float(root[start-1:end+1])
                comb = tuple(path_list[-5:-2])
                if corr == thresh:
                    robustness = pickle.loads(open(os.path.join(root, f), 'rb').read())
                    results[comb]['robustness'] = robustness
            elif re.search(r'(?=^tuning)(?=.*npy$)', f):
                comb = tuple(path_list[-3:])
                tuning_cross_entropies = np.load(os.path.join(root, f))
                if any(np.isnan(tuning_cross_entropies)):
                    nans.append(comb)
                else:
                    results[comb]['tuning_cross_entropies'] = tuning_cross_entropies
                    results[comb]['root'] = root
            elif re.search(r'(?=^pruning)(?=.*npy$)', f):
                comb = tuple(path_list[-3:])
                pruning_cross_entropies = np.load(os.path.join(root, f))
                if any(np.isnan(pruning_cross_entropies)):
                    if not comb in nans:
                        nans.append(comb)
                else:
                    results[comb]['pruning_cross_entropies'] = pruning_cross_entropies
    for nan in nans:
        del results[nan]
    return results


def get_best_comb(results: Dict[tuple, dict]) -> tuple:
    return min({comb : np.mean(metrics['tuning_cross_entropies']) for comb, metrics in results.items()}.items(), key=lambda kv:kv[1])[0]


def del_paths(PATH: str, best_comb: tuple) -> None:
    for root, _, files in os.walk(PATH):
        for f in files:
            if re.search(r'(?=.*entropies)(?=.*npy$)', f): 
                comb = tuple(root.split('/')[-3:])
                if comb != best_comb:
                    try:
                        shutil.rmtree(root)
                    except:
                        print(f'{root} does not exist. It has been deleted before.\n')
                        pass


if __name__ == '__main__':
    args = parseargs()
    results, trees = get_split_results(args.in_path, args.percentages args.thresh)
    
    with open(os.path.join(args.in_path, 'validation_results.json'), 'w') as f:
        json.dump(results, f)
    
    for p, roots in trees.items():
        with open(os.path.join(args.in_path, f'{p:02d}', 'model_paths.txt'), 'w') as f:
            count = 0
            for root in roots:
                for seed in args.seeds:
                    model_file = '/'.join((root, f'seed{seed:02d}', 'model', 'model_epoch1000.tar'))
                    f.write(model_file)
                    f.write('\n')
                    count += 1
            assert count == int((100 // p) * len(args.seeds))

