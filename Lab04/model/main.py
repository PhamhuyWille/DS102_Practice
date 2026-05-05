import pandas as pd
import preprocessing as prp
from model import DecisionTreescratch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def preprocessing():
    red_path = '../data/winequality-red.csv'
    white_path = '../data/winequality-white.csv'

    df_red = prp.pre(red_path, 'red')
    df_white = prp.pre(white_path, 'white')

    df = pd.concat([df_red, df_white], axis=0).fillna(0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop(columns='quality')
    y = df['quality']
    y = y.map({
        1:0, 2:0, 3:0, 4:0, 5:1, 
        6:2, 7:2, 8:2, 9:2, 10:2
    })

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = prp.StandardScaler()
    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
    
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])

    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    return X_train, X_test, y_train, y_test

def run_on_scratch(X_train, X_test, y_train):
    model = DecisionTreescratch()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def run_model(X_train, X_test, y_train):
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f'accuracy: {accuracy}')
    confusion = confusion_matrix(y_true, y_pred)
    print('confusion_matrix:')
    print(confusion)

def main():
    X_train, X_test, y_train, y_test = preprocessing()
    y_pred_scratch = run_on_scratch(X_train, X_test, y_train)
    y_pred_library = run_model(X_train, X_test, y_train)

    metrics(y_test, y_pred_scratch)
    metrics(y_test, y_pred_library)

if __name__ == '__main__':
    main()