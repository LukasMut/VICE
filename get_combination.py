#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from typing import List, Tuple

def create_combinations(
                        slabs: np.ndarray,
                        spikes: np.ndarray,
                        pis: np.ndarray,
) -> List[Tuple[float, float, float]]:
    combinations = []
    for slab in slabs:
        for spike in spikes:
            for pi in pis:
                if (spike < slab):
                    combinations.append((spike, slab, pi))
    return combinations

if __name__ == '__main__':
    idx = int(sys.argv[1])

    slabs = np.power(2., np.arange(-4, 3, 1))
    spikes = np.power(2., np.arange(-4, 3, 1))
    pis = np.arange(0.05, 1.0, 0.05).round(2)

    combinations = create_combinations(slabs, spikes, pis)
    subset = combinations[idx]
    np.savetxt('current_subset.txt', subset)
