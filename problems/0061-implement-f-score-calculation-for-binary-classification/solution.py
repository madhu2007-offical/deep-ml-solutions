
import numpy as np

def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    b2 = beta ** 2
    denominator = (1 + b2) * tp + b2 * fn + fp

    if denominator == 0:
        return 0.0

    return round(float((1 + b2) * tp / denominator), 3)
