import cv2 as cv
import numpy as np
import os
import random
from tqdm import tqdm

def collect(split: str='train', BASE_DIR: str='../data/'):
    '''
    Lấy dữ liệu từ folder data, sau đó thông qua việc lấy tên file của folder
    sẽ lấy hình ảnh và xử lý thông qua thư viện OpenCV. Cuối cùng sẽ lưu lại trong 
    list images và labels tương ứng đối với file đó (1: Normal, -1: Pneimonia).
    Cuối cùng sẽ shuffle lại list images và labels 1 lần để thứ tự ảnh random và 
    trả về X và y tương ứng là images và labels.
    '''
    normal = 'NORMAL'       # 1
    pneu = 'PNEUMONIA'      # -1
    
    images = []
    labels = []

    for img_file in tqdm(os.listdir(os.path.join(BASE_DIR, split, normal))):
        img = cv.imread(os.path.join(BASE_DIR, split, normal, img_file))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT).reshape(-1)
        images.append(img)
        labels.append(1)

    for img_file in tqdm(os.listdir(os.path.join(BASE_DIR, split, pneu))):
        img = cv.imread(os.path.join(BASE_DIR, split, pneu, img_file))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR_EXACT).reshape(-1)
        images.append(img)
        labels.append(-1)

    X = np.stack(images, axis=0)
    y = np.array(labels)
    
    data = list(zip(X, y))
    random.shuffle(data)

    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    return X, y

class Scaler:
    def __init__(self):
        '''
        Khởi tạo các giá trị ban đầu cho lớp scaler, bao gồm
        - self.mean là None
        - self.std là None
        '''
        self.mean = None
        self.std = None

    def fit_transform(self, X: np.ndarray):
        '''
        (Using in training set)
        Tính toán mean và std của X sau đó sẽ chuyển dữ liệu 
        về phối có trung bình bằng 0 và độ lệch chuẩn là 1.
        '''
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X_scaler = (X - self.mean) / self.std
        return X_scaler
    
    def transform(self, X: np.ndarray):
        '''
        (Using for test set) 
        Sử dụng mean và std của tập train để chuyển dữ liệu 
        trong tập test.
        '''
        X_scaler = (X - self.mean) / self.std
        return X_scaler