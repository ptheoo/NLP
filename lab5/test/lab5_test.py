import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset # <-- THƯ VIỆN MỚI
from typing import List

# Thêm đường dẫn src vào hệ thống để import TextClassifier
# Giả định script này chạy từ NLP/lab5/test/
# Cần đảm bảo file src/models/text_classifier.py tồn tại
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'models')))
from src.models.text_classifier import TextClassifier 

# ----------------------------------------------------
# Task 1 & Task 3 Setup: Data Definition (Sử dụng Dataset thực tế)
# ----------------------------------------------------
print("--- Tải Bộ Dữ liệu Mới (Financial Sentiment) ---")

try:
    # Tải bộ dữ liệu và chỉ lấy tập huấn luyện
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split='train')
except ImportError:
    print("LỖI: Cần cài đặt thư viện 'datasets'. Chạy: pip install datasets")
    sys.exit(1)


# Lấy một tập con nhỏ (ví dụ: 1000 mẫu đầu tiên) để quá trình chạy nhanh
SAMPLE_SIZE = 1000
if len(ds) > SAMPLE_SIZE:
    ds = ds.select(range(SAMPLE_SIZE))

# Trích xuất texts và labels
texts: List[str] = ds['text']

# Ánh xạ nhãn 3 lớp sang nhị phân (0=Negative/Neutral, 1=Positive)
# Nhãn gốc: 0 (negative), 1 (neutral), 2 (positive)
labels: List[int] = [1 if sentiment == 2 else 0 for sentiment in ds['label']]

print(f"Tổng số mẫu dữ liệu đã tải và ánh xạ (0/1): {len(texts)}")
print(f"Kiểm tra phân bổ nhãn (Positive count): {sum(labels)}")
print("-" * 40)

# ----------------------------------------------------
# Task 3: Split the data
# ----------------------------------------------------
# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(
    texts, 
    labels, 
    test_size=0.2, # 20% cho tập kiểm tra (200 mẫu)
    random_state=42, 
    stratify=labels # Đảm bảo tỷ lệ nhãn cân bằng
)

print(f"Kích thước tập huấn luyện: {len(X_train)} samples")
print(f"Kích thước tập kiểm tra: {len(X_test)} samples")
print("-" * 40)

# ----------------------------------------------------
# Task 1 Component: Instantiate Vectorizer (Cải tiến)
# ----------------------------------------------------
# Khởi tạo TfidfVectorizer, loại bỏ stop words để tăng hiệu suất
vectorizer = TfidfVectorizer(stop_words='english')

# ----------------------------------------------------
# Task 3: Instantiate and Train Classifier
# ----------------------------------------------------
classifier = TextClassifier(vectorizer=vectorizer)

# Huấn luyện mô hình
classifier.fit(X_train, y_train)

# ----------------------------------------------------
# Task 3: Make Predictions and Evaluate
# ----------------------------------------------------
print("\n--- Dự đoán và Đánh giá (Task 3) ---")

# Dự đoán trên tập kiểm tra
predictions = classifier.predict(X_test)

# In một vài mẫu dự đoán để kiểm tra nhanh
print(f"Kiểm tra 5 dự đoán đầu tiên (True vs Pred):")
for i in range(5):
    print(f"  Thực tế: {y_test[i]} | Dự đoán: {predictions[i]} | Text: {X_test[i][:50]}...")

print("-" * 40)

# Đánh giá các dự đoán
metrics = classifier.evaluate(y_true=y_test, y_pred=predictions)

# In kết quả
print("\n KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ")
for metric, value in metrics.items():
    print(f"- {metric.capitalize()}: {value:.4f}")
print("-" * 40)