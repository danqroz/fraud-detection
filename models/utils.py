import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def estimate_gaussian(X: pd.DataFrame) -> tuple[pd.Series, np.ndarray[float]]:
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)
    return mu, sigma


def _f1_score(y: pd.Series, predictions: np.ndarray[int]) -> float:
    tp = np.sum((y == 1) & (predictions == 1))
    fp = np.sum((y == 0) & (predictions == 1))
    fn = np.sum((y == 1) & (predictions == 0))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    return f1_score


def select_threshold(y: pd.Series, probs: np.ndarray[float]) -> tuple[float, float]:
    best_eps = best_f1 = 0

    epsilons = [
        1.0527717316e-70,
        1.0527717316e-60,
        1.0527717316e-50,
        1.0527717316e-40,
        1.0527717316e-30,
        1.0527717316e-24,
    ]
    for epsilon in epsilons + list(np.linspace(1.01 * min(probs), max(probs), 1000)):
        predictions = (probs < epsilon).astype(int)

        f1 = _f1_score(y, predictions)

        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon
    return best_eps, best_f1


def multivariate_gaussian(
    X: pd.DataFrame, mu: pd.Series, sigma: np.ndarray[float]
) -> np.ndarray[float]:
    prob = multivariate_normal(mean=mu, cov=sigma)

    return prob.pdf(X)
