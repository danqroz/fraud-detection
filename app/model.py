import os

import pandas as pd

from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

load_dotenv()

df = pd.read_csv(os.getenv("DATA_PATH"))
df.drop("Time", axis=1, inplace=True)
target = "Class"

X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

scores = {
    "f1 score": f1_score(y_test, predictions),
    "recall": recall_score(y_test, predictions),
    "precision_score": precision_score(y_test, predictions),
    "auc": roc_auc_score(y_test, predictions),
}
