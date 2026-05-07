# Báo cáo Lab04: Xây dựng mô hình Decision Tree và Random Forest bằng Numpy

---

## 1. Giới thiệu

Trong bài thực hành này, tôi tập trung triển khai hai thuật toán nền tảng của học máy có giám sát là **Decision Tree** (Cây quyết định) và **Random Forest** (Rừng ngẫu nhiên) từ con số không (From Scratch).

**Mục tiêu chính:**

- Hiểu sâu cơ chế toán học và logic phân tách dữ liệu của cây quyết định.
- Nắm vững kỹ thuật **Ensemble Learning** (Học kết hợp) thông qua Random Forest.
- So sánh trực quan hiệu suất giữa mô hình tự xây dựng và thư viện chuẩn `scikit-learn`.

---

## 2. Cấu trúc thư mục

```text
\Lab04
├─ data
│  ├─ winequality-red.csv         # Dữ liệu rượu vang đỏ
│  ├─ winequality-white.csv       # Dữ liệu rượu vang trắng
│  └─ winequality.names
├─ modul                          # Chứa các module xử lý chính
│  ├─ dtmodel.py                  # Mô hình Decision Tree từ scratch
│  ├─ rfmodel.py                  # Mô hình Random Forest từ scratch
│  ├─ preprocessing.py            # Tiền xử lý dữ liệu & Scaling
│  ├─ run_model.py                # Wrapper chạy mô hình & tính Metrics
│  └─ main.py                     # Luồng thực thi chính
└─ report.md                      # Báo cáo thực hành
```

**Cách vận hành:**

```bash
cd Lab04/model
python main.py
```

---

## 3. Các mô hình triển khai

### 3.1. Decision Tree (Cây quyết định)

Mô hình sử dụng tiêu chí **Gini Impurity** để đo lường độ vẩn đục của dữ liệu và tìm ra ngưỡng chia (threshold) tối ưu cho từng nút.

**Quy trình thực hiện:**

1. **Duyệt Feature:** Xem xét mọi đặc trưng đầu vào để tìm khả năng phân loại.
2. **Thử ngưỡng (Threshold):** Duyệt qua các giá trị thực tế của đặc trưng để tìm điểm chia tốt nhất.
3. **Tính toán Gini:** Áp dụng công thức $G = 1 - \sum p_i^2$.
4. **Tối ưu hóa:** Chọn cặp (Feature, Threshold) có tổng chỉ số Gini của các nút con là thấp nhất.
5. **Đệ quy:** Xây dựng cây cho đến khi đạt độ sâu tối đa hoặc các nút lá đã thuần khiết.

### 3.2. Random Forest (Rừng ngẫu nhiên)

Triển khai theo phương pháp **Bagging** để giảm thiểu hiện tượng Overfitting và tăng tính ổn định.

**Kỹ thuật mấu chốt:**

- **Bootstrapping:** Lấy mẫu dữ liệu có thay thế (tạo ra các tập train khác nhau cho mỗi cây).
- **Majority Voting:** Tổng hợp kết quả dự đoán từ tất cả các cây thành phần thông qua cơ chế bầu chọn đa số.

---

## 4. Quy trình Tiền xử lý dữ liệu (Preprocessing)

Quy trình được thiết kế đồng bộ để đảm bảo dữ liệu đầu vào khách quan cho cả mô hình tự viết và thư viện:

- **Tích hợp:** Kết hợp hai tập dữ liệu Red và White, bổ sung biến định danh `is_red` và `is_white`.
- **Label Mapping:** Quy hoạch nhãn chất lượng về 3 nhóm: **Thấp (0)**, **Trung bình (1)**, và **Cao (2)**.
- **Standardization:** Tự triển khai `StandardScaler` để chuẩn hóa các đặc trưng số về phân phối chuẩn (Mean=0, Std=1).
- **Visualization:** Tích hợp thanh tiến trình `tqdm` để giám sát quá trình xử lý dữ liệu theo thời gian thực.

---

## 5. Đánh giá & Kết quả

Kết quả thu được sau khi chạy trên tập dữ liệu kiểm tra (Test set):

<div align="center">

| Chỉ số               | DT Scratch | DT Library | RF Scratch | RF Library |
| :------------------- | :--------: | :--------: | :--------: | :--------: |
| **Accuracy**         |   0.6069   |   0.6054   |   0.6585   | **0.6638** |
| **F1-Score (Macro)** |   0.5918   |   0.5918   | **0.6425** |   0.6395   |

</div>

**Nhận xét:**

- **Độ chính xác:** Mô hình Scratch đạt hiệu suất tương đương với `scikit-learn`, cho thấy logic tính toán Gini và đệ quy xây dựng cây đã được triển khai chính xác.
- **Sức mạnh Ensemble:** Random Forest cải thiện độ chính xác khoảng **5-6%** so với Decision Tree, minh chứng cho hiệu quả của việc giảm phương sai (variance) trong mô hình.
- **Khả năng cân bằng:** RF Scratch đạt F1-Score cao nhất (**0.6425**), cho thấy mô hình tự viết có khả năng phân loại tốt đồng đều trên cả 3 nhóm chất lượng rượu.

---

## 6. Kết luận

1. **Về kiến thức:** Việc tự triển khai giúp làm chủ hoàn toàn các tham số nội tại và hiểu rõ cách dữ liệu được phân tách.
2. **Về kỹ năng:** Hoàn thiện tư duy lập trình hướng module, giúp mã nguồn sạch sẽ, dễ bảo trì và tái sử dụng.
3. **Bài học:** Mặc dù kết quả dự đoán của mô hình Scratch rất tốt, nhưng tốc độ huấn luyện của thư viện `sklearn` vẫn nhanh hơn nhờ được tối ưu hóa sâu ở tầng hệ thống.
