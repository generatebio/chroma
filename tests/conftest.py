import random

try:
    import numpy as np

    HAS_NP = True
except Exception:
    HAS_NP = False

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def pytest_runtest_setup(item):
    # Fix seeds at the beginning of each test.
    seed = 20220714
    random.seed(seed)
    if HAS_NP:
        np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
