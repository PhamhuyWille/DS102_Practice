from preprocessing import collect, scaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def main():
    X_train, y_train = collect(split='train')
    X_test, y_test = collect(split='test')

    X_train, mean, std = scaler(X_train)
    X_test = (X_test - mean)/std

    # Model
    model = SVC(kernel='linear', C=10e-2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f'F1 score: {f1}')
    print(f'Precision score: {precision}')
    print(f'Recall score: {recall}')

    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()