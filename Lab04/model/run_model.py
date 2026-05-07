from rfmodel import RandomForest
from dtmodel import DecisionTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

def metrics(y_true, y_pred, model_name):
    '''
    Hàm metrics dùng để đánh giá các mô hình thông qua các thông số
    accuracy, f1_score, confusion_matrix và classification_report.
    '''
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"\n{'='*20} {model_name} {'='*20}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"\nClassification Report:\n{report}")
    
    return [accuracy, f1]

def DTscratch(X_train, X_test, y_train, y_test):
    '''
    Hàm DTscratch dùng để huấn huyện mô hình Decision Tree được xây
    dựng chỉ sử dụng numpy, sau đó sẽ đánh giá mô hình thông qua hàm
    metrics.
    '''
    print("Training Decision Tree (Scratch)...")
    model = DecisionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics(y_test, y_pred, "Decision Tree (Scratch)")

def DTlib(X_train, X_test, y_train, y_test):
    '''
    Hàm DTlib dùng để huấn luyện mô hình Decision Tree được xây dựng
    từ thư viện sklearn, sau đó sẽ đánh giá mô hình thông qua hàm
    metrics. Kết quả của mô hình dùng để so sánh với kết quả của mô
    hình được xây dựng từ numpy.
    '''
    print("Training Decision Tree (Library)...")
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics(y_test, y_pred, "Decision Tree (Library)")

def RFscratch(X_train, X_test, y_train, y_test):
    '''
    Hàm RFscartch dùng để huấn luyện mô hình Random Forest được xây
    dựng chỉ sử dụng numpy, sau đó sẽ thực hiện đánh giá kết quá của
    mô hình thông qua hàm metrics.
    '''
    print("Training Random Forest (Scratch)...")
    model = RandomForest(n_estimators=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics(y_test, y_pred, "Random Forest (Scratch)")

def RFlib(X_train, X_test, y_train, y_test):
    '''
    Hàm RFlib dùng để huấn luyện mô hình Random Forest được gọi từ
    thư viện sklearn, sau đó thực hiện đánh giá kết quả dự đoán thông
    qua hàm metrics. Kết quả này dùng để so sánh với mô hình được
    xây dựng bằng numpy.
    '''
    print("Training Random Forest (Library)...")
    model = RandomForestClassifier(n_estimators=20, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metrics(y_test, y_pred, "Random Forest (Library)")