
import numpy as np
from itertools import product

def polynomial_basis(s, n):
    return np.array([
        np.prod(np.power(s, c))
        for c in product(range(n + 1), repeat=len(s))
    ], dtype=float)


def gradient_mc_poly(samples, n, alpha, n_passes):
    k = len(samples[0][0])
    w = np.zeros((n + 1) ** k)

    for _ in range(n_passes):
        for s, G in samples:
            x = polynomial_basis(s, n)
            v = np.dot(w, x)
            w += alpha * (G - v) * x

    return w
