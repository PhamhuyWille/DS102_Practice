import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class StandardScaler:
    '''
    Lớp StandardScaler được sử dụng để chuẩn hóa dữ liệu theo phân phối 
    chuẩn với phương sai là 0 và variance là 1.
    '''
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
    '''
    Hàm pre dùng để đọc file và thêm thuộc tính is_label.
    Sau đó sẽ trả về một DataFrame.
    '''
    df = pd.read_csv(path, sep=';')
    df[f'is_{label}'] = True
    return df


def preprocessing():
    '''
    Hàm preprocessing được sử dụng để chuẩn bị dữ liệu cho huấn luyện mô hình. 
    Pipeline bao gồm:
    1. Đọc dữ liệu
    2. Merge dữ liệu thành 1 DataFrame
    3. Tách dữ liệu thành X và y
    4. Chia dữ liệu thành training set và test set
    5. Chuẩn hóa dữ liệu
    Hàm này sẽ trả về X_train, X_test, y_train và y_test đã được chuẩn hóa.
    '''
    pbar = tqdm(total=5, desc="Processing Data", unit="step")

    red_path = '../data/winequality-red.csv'
    white_path = '../data/winequality-white.csv'

    print(f"Loading data from {red_path} and {white_path}...")
    df_red = pre(red_path, 'red')
    df_white = pre(white_path, 'white')
    pbar.update(1)

    print("Merging and shuffling datasets...")
    df = pd.concat([df_red, df_white], axis=0).fillna(0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    pbar.update(1)

    print("Mapping quality labels and splitting features/target...")
    X = df.drop(columns='quality')
    y = df['quality']
    y = y.map({
        1:0, 2:0, 3:0, 4:0, 5:0, 
        6:1, 
        7:2, 8:2, 9:2, 10:2
    })
    pbar.update(1)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pbar.update(1)

    print("Scaling numerical features...")
    scaler = StandardScaler()
    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
    
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])

    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    pbar.update(1)
    pbar.close()
    print("Preprocessing completed successfully.\n")
    
    return X_train, X_test, y_train, y_test