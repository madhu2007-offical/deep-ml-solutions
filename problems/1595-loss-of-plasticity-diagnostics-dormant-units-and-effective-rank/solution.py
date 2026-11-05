
import numpy as np

def dormant_ratio(H, tau):
    H = np.asarray(H, dtype=float)
    means = np.abs(H).mean(axis=0)
    normalizer = means.mean()

    if normalizer == 0:
        scores = np.zeros_like(means)
    else:
        scores = means / normalizer

    indices = np.where(scores <= tau)[0].tolist()
    return float(len(indices) / H.shape[1]), indices


def effective_rank(H, eps=1e-12):
    H = np.asarray(H, dtype=float)
    Hc = H - H.mean(axis=0)

    s = np.linalg.svd(Hc, compute_uv=False)
    s = s[s >= eps]

    if s.size == 0:
        return 0.0

    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def plasticity_report(H, W, tau, decimals=4):
    ratio, _ = dormant_ratio(H, tau)

    return {
        "dormant_ratio": round(ratio, decimals),
        "effective_rank": round(effective_rank(H), decimals),
        "weight_norm": round(float(np.linalg.norm(W, ord="fro")), decimals)
    }
