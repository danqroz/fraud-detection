import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()


class SplitData:
    PATH = os.getenv("DATA_PATH")
    TARGET = "Class"
    KEY_VARS = ["Time"]
    ALL_COLS = [f"V{i}" for i in range(1, 29)]
    ALL_COLS += ["Amount"]

    def __init__(self, columns=ALL_COLS) -> None:
        df = pd.read_csv(self.PATH, usecols=lambda col: col in columns)
        self.X = df.drop(self.TARGET, axis=1)
        self.y = df[self.TARGET]

    def random_split(self, test_size: float = 0.15):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )

        return X_train, X_test, y_train, y_test

    def anomaly_split(self, train_frac: float = 0.8, test_size: float = 0.5):

        df_train = pd.concat([self.X, self.y], axis=1)[self.y == 0].sample(
            frac=train_frac
        )
        X_train = df_train.drop(self.TARGET, axis=1)
        y_train = df_train[self.TARGET]

        train_idx = self.y.index.isin(y_train.index)

        X_val_test = self.X[~train_idx]
        y_val_test = self.y[~train_idx]

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=test_size,
            random_state=42,
            stratify=y_val_test,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
