import numpy as np

def bincount(a: np.ndarray, n_bins: int) -> np.ndarray:
    c = np.zeros(n_bins, dtype=int)
    np.add.at(c, a, 1)
    return c