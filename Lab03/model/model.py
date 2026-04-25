import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

class SVM:
    def __init__(self, C: float = 1, lr: float = 0.01, epochs: int = 100, batch_size: int = 32):
        '''
        Định nghĩa các thông số của mô hình, bao gồm các thông số sau:
        - lr: learning rate (tốc độ học của mô hình)
        - epochs: Số lượng vòng lặp để mô hình hội tụ
        - batch_size: Số lượng mẫu được sử dụng trong 1 lần để cập nhật thông số.
        - C: Điều khiển trade off giữa outlier và độ rộng margin
        - self.losses: list sai số của mô hình để vẽ đồ thị biểu diễn.
        '''
        self.weight = None
        self.bias = None
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Được sử dụng để training mô hình. Mô hình sử dụng Gradient Descent
        với Hinge Loss để giúp mô hình hội tụ. Đồng thời, sử dụng mini_batch 
        thay vì SGD để giảm khối lượng tính toán. 
        Quy trình:
        - Đầu tiên khởi tạo self.weight và self.bias. Sau đó, với mỗi epoch
        thì bắt đầu shuffle X và y để mô hình tránh học theo thứ tự cố định.
        - Tiếp theo, mô hình sẽ học thông qua từng batch_size và tính GD để 
        tinh chỉnh weight và bias. 
        - Hết vòng lặp thì mô hình sẽ dự đoán kết quả và tính toán loss thông qua
        loss_fn và lưu vào self.losses để vẽ đồ thị sau này. 
        '''
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
        '''
        Dùng để dự đoán kết quả trên tập test.
        '''
        y_score = X @ self.weight + self.bias
        return np.where(y_score >= 0, 1, -1)

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray):
        '''
        Hàm tính loss sử dụng Hinge Loss để vẽ đồ thị sau này.
        '''
        reg = 0.5 * np.dot(self.weight, self.weight)
        hinge_loss = self.C * np.maximum(0, 1 - y * y_hat).sum()
        return reg + hinge_loss
    
    def loss(self):
        '''
        Sau khi training xong, sẽ sử dụng hàm loss để vẽ sai số thông qua 
        những lần training.
        '''
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