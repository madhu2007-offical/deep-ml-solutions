
import numpy as np

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)
        self.log_prior = []
        self.log_prob = []
        self.log_inv_prob = []

        for c in self.classes:
            X_c = X[y == c]
            n = len(X_c)

            prior = n / len(y)
            prob = (X_c.sum(axis=0) + self.smoothing) / (
                n + 2 * self.smoothing
            )

            prob = np.clip(prob, 1e-15, 1 - 1e-15)

            self.log_prior.append(np.log(prior))
            self.log_prob.append(np.log(prob))
            self.log_inv_prob.append(np.log(1 - prob))

        self.log_prior = np.array(self.log_prior)
        self.log_prob = np.array(self.log_prob)
        self.log_inv_prob = np.array(self.log_inv_prob)

    def predict(self, X):
        X = np.asarray(X)

        scores = (
            X @ self.log_prob.T
            + (1 - X) @ self.log_inv_prob.T
            + self.log_prior
        )

        return self.classes[np.argmax(scores, axis=1)]
