import numpy as np

def laplace_logreg(X, y, prior_var, n_iter=50):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    D = X.shape[1]
    w = np.zeros(D)

    for _ in range(n_iter):
        z = X @ w
        mu = 1.0 / (1.0 + np.exp(-z))

        gradient = X.T @ (y - mu) - w / prior_var

        S = mu * (1.0 - mu)
        H = X.T @ (X * S[:, None]) + np.eye(D) / prior_var

        step = np.linalg.solve(H, gradient)
        w += step

        if np.linalg.norm(step) < 1e-12:
            break

    z = X @ w
    mu = 1.0 / (1.0 + np.exp(-z))
    S = mu * (1.0 - mu)

    H = X.T @ (X * S[:, None]) + np.eye(D) / prior_var
    Sigma = np.linalg.inv(H)

    return w, Sigma


def predictive(X_test, w_map, Sigma, method, n_samples=1000, seed=0):
    X_test = np.asarray(X_test, dtype=float)
    w_map = np.asarray(w_map, dtype=float)

    m = X_test @ w_map

    if method == "plugin":
        return 1.0 / (1.0 + np.exp(-m))

    elif method == "mc":
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(Sigma)

        Z = rng.standard_normal((n_samples, len(w_map)))
        W = w_map + Z @ L.T

        probabilities = 1.0 / (1.0 + np.exp(-(X_test @ W.T)))

        return np.mean(probabilities, axis=1)

    elif method == "probit":
        v = np.sum((X_test @ Sigma) * X_test, axis=1)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * v / 8.0)

        return 1.0 / (1.0 + np.exp(-(kappa * m)))

    else:
        raise ValueError("Unknown prediction method")