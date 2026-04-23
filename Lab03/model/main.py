from tqdm import tqdm
from preprocessing import collect
from model import SVM

def main():
    BASE_DIR = '../data/'
    X_train, y_train = collect('train')
    X_test, y_test = collect('test')
    
    model = SVM(epochs=1000, lr=0.001, C=10e-2, batch_size=64)
    model.fit(X_train, y_train)
    print(model.metric(X_test, y_test))
    model.loss()

if __name__ == '__main__':
    main()