import utils
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from split_data import SplitData

split = SplitData()
X_train, X_val, X_test, y_train, y_val, y_test = split.anomaly_split()

mu, sigma = utils.estimate_gaussian(X_train)
probs = utils.multivariate_gaussian(X_val, mu, sigma)
epsilon, f1 = utils.select_threshold(y_val, probs)

probs_test = utils.multivariate_gaussian(X_test, mu, sigma)
predictions = probs_test < epsilon

scores = {
    "f1 score": f1_score(y_test, predictions),
    "recall": recall_score(y_test, predictions),
    "precision_score": precision_score(y_test, predictions),
    "auc": roc_auc_score(y_test, predictions),
}

reduced_cols = [
    "V1",
    "V3",
    "V4",
    "V7",
    "V9",
    "V10",
    "V11",
    "V12",
    "V14",
    "V16",
    "V17",
    "V18",
]
