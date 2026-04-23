import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SVM:
    def __init__(self, C: float = 1, lr: float = 0.01, epochs: int = 1000, batch_size: int = 32):
        self.weight = None
        self.bias = None
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, dim = X.shape
        self.weight = np.zeros(dim)
        self.bias = 0
        # weight (136384, 1)
        for epoch in tqdm(range(self.epochs)):
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, N, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                # X_batch (64, 136384)
                # y_batch (64, 1)
                y_pred =  X_batch @ self.weight.T + self.bias
                # Take care of case 2: yn * an < 1
                mask = (y_batch * y_pred) < 1
                dw = self.weight - self.C * np.dot(y_batch[mask], X_batch[mask])
                db = -self.C * np.sum(y_batch[mask])
                
                self.weight -= self.lr * (dw / self.batch_size)
                self.bias -= self.lr * (db / self.batch_size)

            y_epoch = self.predict(X)
            # y_epoch (5216, 1)
            # y (5126, 1)
            loss = self.loss_fn(y, y_epoch)
            self.losses.append(loss)

    def predict(self, X: np.ndarray):
        y_score = X @ self.weight + self.bias
        return np.where(y_score >= 0, 1, -1)

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray):
        reg = 0.5 * np.dot(self.weight, self.weight)
        hinge_loss = self.C * np.maximum(0, 1 - y * y_hat).sum()
        return reg + hinge_loss
    
    def metric(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        
        precision = precision_score(y_pred, y)
        recall = recall_score(y_pred, y)
        f1 = f1_score(y_pred, y)

        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1
        }
    
    def loss(self):
        df = pd.DataFrame({
            'Epoch': range(1, len(self.losses) + 1),
            'Loss': self.losses
        })

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Epoch', y='Loss', color='teal', linewidth=2.5)

        plt.title('The loss function by each epoch', fontsize=15)
        plt.xlabel("The number of Epoch", fontsize=12)
        plt.ylabel("Hinge Loss", fontsize=12)
        plt.savefig('../Loss.png', dpi=300, bbox_inches='tight')
        plt.show()
