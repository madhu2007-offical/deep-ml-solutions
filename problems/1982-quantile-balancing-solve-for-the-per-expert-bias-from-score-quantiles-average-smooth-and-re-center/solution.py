import numpy as np


def qb_route(scores, beta, k):
    scores = np.asarray(scores, dtype=float)
    beta = np.asarray(beta, dtype=float)
    T, N = scores.shape

    if (T * k) % N != 0:
        raise ValueError(f"T * k ({T * k}) must be divisible by N ({N})")
    C = (T * k) // N

    # Biased scores: s_{t, e} - beta_e
    biased = scores - beta[None, :]

    # Sort descending per token; mergesort preserves lower index first on ties
    order = np.argsort(-biased, axis=1, kind="mergesort")
    idx = order[:, :k]

    # (k+1)-th largest biased score per token (runner-up threshold a_t)
    sorted_biased = np.take_along_axis(biased, order, axis=1)
    a_t = sorted_biased[:, k]

    # Margins s_{t, e} - a_t; (C+1)-th largest over tokens per expert
    margins = scores - a_t[:, None]
    sorted_margins = np.sort(margins, axis=0)[::-1]
    beta_local = sorted_margins[C, :]

    return idx, beta_local


def qb_train(steps, k, ema=0.0):
    def r4(arr):
        return (np.round(np.asarray(arr, dtype=float), 4) + 0.0).tolist()

    def maxvio(counts):
        c_mean = np.mean(counts)
        return float(np.max(counts) / c_mean - 1.0)

    N = np.asarray(steps[0][0]).shape[1]
    beta = np.zeros(N, dtype=float)

    global_maxvios = []
    micro_maxvios = []

    for global_batch in steps:
        global_counts = np.zeros(N, dtype=float)
        micro_vios = []
        beta_locals = []

        for micro_scores in global_batch:
            idx, beta_local = qb_route(micro_scores, beta, k)
            counts = np.bincount(idx.ravel(), minlength=N).astype(float)

            global_counts += counts
            micro_vios.append(maxvio(counts))
            beta_locals.append(beta_local)

        global_maxvios.append(r4(maxvio(global_counts)))
        micro_maxvios.append(r4(np.mean(micro_vios)))

        beta_bar = np.mean(beta_locals, axis=0)
        beta = ema * beta + (1.0 - ema) * beta_bar
        beta = beta - np.mean(beta)

    return {
        "global_maxvio": global_maxvios,
        "micro_maxvio": micro_maxvios,
        "beta": r4(beta),
    }