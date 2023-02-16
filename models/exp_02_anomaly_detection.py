import utils
from sklearn.metrics import f1_score, precision_score, recall_score
from split_data import SplitData

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
    "Class"
]

split = SplitData(reduced_cols)
X_train, X_val, X_test, y_train, y_val, y_test = split.anomaly_split()

# Generate Gaussian
mu, sigma = utils.estimate_gaussian(X_train)
# Select threshold based on f1 score
probs = utils.multivariate_gaussian(X_val, mu, sigma)
epsilon, f1 = utils.select_threshold(y_val, probs)
# make predictions
probs_test = utils.multivariate_gaussian(X_test, mu, sigma)
predictions = probs_test < epsilon

scores = {
    "f1 score": f1_score(y_test, predictions),
    "recall": recall_score(y_test, predictions),
    "precision_score": precision_score(y_test, predictions),
}
print(scores)
