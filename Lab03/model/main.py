import numpy as np
from tqdm import tqdm
from preprocessing import collect, Scaler
from model import SVM
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

def metric(y, y_pred):
    '''
    Các metric được sử dụng trong bài thực hành này là:
    - precision_score
    - recall_score
    - f1_score
    '''
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')

def library(X_train, y_train, X_test, y_test):
    '''
    Hàm này sẽ gọi ra mô hình trong thư viện scikit-learn
    và tính toán metric trên hàm metric
    '''
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Metric from Standard Library:')
    metric(y_test, y_pred)

def from_scratch(X_train, y_train, X_test, y_test):
    '''
    Hàm này sẽ gọi ra mô hình from scratch với epochs là 1000
    và lr là 0.001 để mô hình hội tụ sâu hơn. Sau đó, kết quả 
    mô hình sẽ được đưa vào hàm metric để tính toán. Cuối cùng,
    loss sau từng epoch sẽ được hiển thị thông qua hàm loss().
    '''
    model = SVM(epochs=1000, lr=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Metric from Scratch:')
    metric(y_test, y_pred)
    model.loss()

def main():
    BASE_DIR = '../data/'
    X_train, y_train = collect('train')
    X_test, y_test = collect('test')
    
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from_scratch(X_train, y_train, X_test, y_test)
    library(X_train, y_train, X_test, y_test)
    
if __name__ == '__main__':
    main()