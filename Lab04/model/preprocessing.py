import pandas as pd
import numpy as np

class StandardScaler:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1 
        X_scaled = (X - self.mean) / self.std
        return X_scaled

    def transform(self, X: np.ndarray):
        X_scaled = (X - self.mean) / self.std
        return X_scaled

def pre(path: str, label: str = 'red'):
    df = pd.read_csv(path, sep=';')
    df[f'is_{label}'] = True
    return df