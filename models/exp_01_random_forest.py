import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from split_data import SplitData
import joblib


def score_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    scores = {
        "f1 score": f1_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
    }
    return scores


split = SplitData()
X_train, X_test, y_train, y_test = split.random_split()

model = RandomForestClassifier()
model.fit(X_train, y_train)

scores = score_model(model, X_test, y_test)
# ['V1', 'V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
importance_df = (
    pd.DataFrame(model.feature_importances_, index=X_train.columns)
    .rename(columns={0: "importances"})
    .sort_values(by="importances", ascending=False)
)

n_estimators = [x for x in range(100, 2000, 200)]
max_features = ["auto", "sqrt"]
max_depth = [x for x in range(10, 110, 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)
rf_random.fit(X_train, y_train)

best_random = rf_random.best_estimator_

random_scores = score_model(best_random, X_test, y_test)

grid_params = {
    "n_estimators": [1000, 1800, 1900, 2000],
    "min_samples_split": [1, 2, 3],
    "min_samples_leaf": [1, 2, 3],
    "max_features": [2, 3],
    "max_depth": [80, 90, 100, 110],
    "bootstrap": [False],
}

rf_ = RandomForestClassifier()

grid_search = GridSearchCV(
    estimator=rf_, param_grid=grid_params, cv=3, n_jobs=-1, verbose=2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_score = score_model(best_model, X_test, y_test)

joblib.dump(model, "./models_save/random_forest.sav")
