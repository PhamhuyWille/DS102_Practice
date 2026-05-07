import numpy as np
from dtmodel import DecisionTree

class RandomForest:
    def __init__(
            self, 
            n_estimators: int = 100, 
            depth: int = 10, 
            min_sample_split: int=2, 
            n_samples: int = None
        ):
        '''
        Hàm init dùng để khởi tạo tham số của mô hình Random Forest, bao gồm:
        - n_estimators: là số lượng cây mà sẽ được tạo ra
        - depth: độ sâu của từng cây.
        - min_sample_split: số lượng mẫu tối thiểu để chia cây
        - n_samples: tổng sample trong training set
        - features: số lượng đặc trưng của dataset
        - models: để lưu trữ lại các cây của mô hình.
        '''
        self.n_estimators = n_estimators
        self.depth = depth
        self.min_sample_split = min_sample_split
        self.n_samples = n_samples
        self.total_sample = None
        self.features = None
        self.models = []

    def bootstrap_sampling(self, X, y):
        '''
        Hàm boostrap_sampling dùng để chia dữ liệu thành những mẫu nhỏ để
        dữ liệu đầu vào của mỗi cây có sự khác biệt, để các cây sẽ học được 
        những lớp ẩn của dữ liệu. Trả về một dataset để training cho 
        mô hình.
        '''
        index = np.random.choice(self.total_sample, size=self.n_samples, replace=True)
        X_sample = X[index]
        y_sample = y[index]
        return X_sample, y_sample

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Hàm fit dùng để huấn luyện mô hình Random Forest thông qua việc 
        lấy 
        '''
        self.total_sample, self.features = X.shape
        if self.n_samples is None:
            self.n_samples = self.total_sample
    
        for i in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sampling(X, y)
            model = DecisionTree(depth=self.depth, min_sample_split=self.min_sample_split)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X: np.ndarray):
        '''
        Hàm predict để dữ đoán class của các sample thông qua gọi hàm
        predict từng mô hình trong Random Forest. Sau đó sẽ thực hiện
        major_vote để lấy kết quả và trả về một mảng các dự đoán.
        '''
        preds = []
        for i in range(self.n_estimators):
            pred_i = self.models[i].predict(X)
            preds.append(pred_i)

        preds = np.array(preds)
        preds = preds.T
        final_predictions = []
        for sample_prediction in preds:
            y_pred = self.major_vote(sample_prediction)
            final_predictions.append(y_pred)
        return np.array(final_predictions)
    
    def major_vote(self, y):
        '''
        Hàm major_vote dùng để lấy chọn nhãn nào có tỉ lệ cao nhất. Hàm
        trả về lớp có tỉ lệ nhiều nhất.
        '''
        labels, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        value = labels[index]
        return value