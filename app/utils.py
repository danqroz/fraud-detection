import numpy as np
from scipy.stats import multivariate_normal


def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.mean(np.square(X - mu), axis=0)
    return mu, sigma2


def _f1_score(y, predictions):
    tp = np.sum((y == 1) & (predictions == 1))
    fp = np.sum((y == 0) & (predictions == 1))
    fn = np.sum((y == 1) & (predictions == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


def select_threshold(y, probs):
    best_eps = best_f1 = 0

    for epsilon in np.linspace(1.01 * min(probs), max(probs), 1000):
        predictions = (probs < epsilon).astype(int)

        f1 = _f1_score(y, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon
    return best_eps, best_f1


def multivariate_gaussian(X, mu, sigma2):
    prob = multivariate_normal(mean=mu, cov=sigma2)
    return prob.pdf(X)
