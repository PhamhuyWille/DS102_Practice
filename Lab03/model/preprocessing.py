import cv2 as cv
import numpy as np
import os
import random
from tqdm import tqdm

def collect(split: str='train', BASE_DIR: str='../data/'):
    normal = 'NORMAL'       # 1
    pneu = 'PNEUMONIA'      #-1
    
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
    X = (X - X.mean()) / X.std()
    y = np.array(labels)
    
    data = list(zip(X, y))
    random.shuffle(data)

    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)
    return X, y