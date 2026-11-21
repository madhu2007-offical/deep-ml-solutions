import numpy as np


def gae_encode(A, X, W1, W2):
    A_tilde = A + np.eye(A.shape[0], dtype=A.dtype)
    deg = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    A_hat = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]
    
    H1 = np.maximum(0.0, A_hat @ X @ W1)
    Z = A_hat @ H1 @ W2
    return Z


def gae_logits(Z, pairs):
    pairs_arr = np.asarray(pairs, dtype=int)
    if pairs_arr.size == 0:
        return np.empty(0, dtype=Z.dtype)
    return np.sum(Z[pairs_arr[:, 0]] * Z[pairs_arr[:, 1]], axis=1)


def _bce_with_logits(logits, targets, pos_weight=1.0):
    log_weight = 1.0 + (pos_weight - 1.0) * targets
    relu_logits = np.maximum(logits, 0.0)
    neg_abs_logits = -np.abs(logits)
    loss = log_weight * (relu_logits + np.log1p(np.exp(neg_abs_logits))) - pos_weight * logits * targets
    return loss


def gae_loss(A, Z, pos_weight=None):
    logits = Z @ Z.T
    if pos_weight is None:
        pos_sum = np.sum(A)
        total_entries = A.size
        pos_weight = (total_entries - pos_sum) / pos_sum
    
    loss = _bce_with_logits(logits, A, pos_weight=pos_weight)
    return float(np.mean(loss))


def sampled_gae_loss(Z, pos_pairs, neg_pairs):
    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)
    
    logits = np.concatenate([pos_logits, neg_logits])
    targets = np.concatenate([
        np.ones(len(pos_logits), dtype=Z.dtype),
        np.zeros(len(neg_logits), dtype=Z.dtype)
    ])
    
    loss = _bce_with_logits(logits, targets, pos_weight=1.0)
    return float(np.mean(loss))


def vgae_sample(mu, logvar, eps):
    std = np.exp(0.5 * logvar)
    return mu + std * eps


def vgae_kl(mu, logvar):
    kl_per_node = -0.5 * np.sum(1.0 + logvar - np.square(mu) - np.exp(logvar), axis=1)
    return float(np.mean(kl_per_node))