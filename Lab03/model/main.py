from tqdm import tqdm
from preprocessing import collect, scaler
from model import SVM

def main():
    BASE_DIR = '../data/'
    X_train, y_train = collect('train')
    X_test, y_test = collect('test')
    
    X_train, mean, std = scaler(X_train)
    X_test = (X_test - mean)/std

    model = SVM(epochs=1000, lr=0.001, C=1, batch_size=32)
    model.fit(X_train, y_train)
    print(model.metric(X_test, y_test))
    model.loss()

if __name__ == '__main__':
    main()