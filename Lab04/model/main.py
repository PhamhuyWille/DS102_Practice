import pandas as pd
import preprocessing as prp
from run_model import DTlib, DTscratch, RFlib, RFscratch
from tqdm import tqdm

def main():
    print("=== KHỞI CHẠY HỆ THỐNG SO SÁNH MÔ HÌNH ===\n")
    # Bước 1: Chuẩn bị dữ liệu và các thông số cho kết quả
    X_train, X_test, y_train, y_test = prp.preprocessing()

    results = []
    models = [
        ("Decision Tree (Scratch)", DTscratch),
        ("Decision Tree (Library)", DTlib),
        ("Random Forest (Scratch)", RFscratch),
        ("Random Forest (Library)", RFlib)
    ]

    # Bước 2: Huấn luyện lần lượt các mô hình DT scratch, DT library, RF scartch, RF library
    print("Đang bắt đầu quá trình huấn luyện và đánh giá...")
    for name, model_func in tqdm(models, desc="Tổng tiến trình huấn luyện", unit="model"):
        result = model_func(X_train, X_test, y_train, y_test)
        results.append(result)

    model_names = [m[0] for m in models]
    metric = pd.DataFrame(
        results, 
        columns=['Accuracy', 'F1-Score'],
        index=model_names
    )

    # Bước 3: Minh họa kết quả đánh giá
    print("\n" + "="*50)
    print("             BẢNG SO SÁNH HIỆU SUẤT")
    print("="*50)
    print(metric)
    print("="*50)
    print("\nQuá trình hoàn tất!")


if __name__ == '__main__':
    main()