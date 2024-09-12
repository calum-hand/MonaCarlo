import os

import numpy as np

def build_sampler():
    seed = int(os.getenv("RANDOM_SEED", 0))
    return np.random.RandomState(seed)

random_sampler = build_sampler()
