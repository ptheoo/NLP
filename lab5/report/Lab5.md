# Báo Cáo Lab 4

## I. Tổng quan và Hướng dẫn Thực thi (Task 1, 2, 3)

### 1. Mô tả Triển khai và Hướng dẫn Thực thi

Dự án này triển khai hệ thống phân loại văn bản nhị phân trên hai framework: Scikit-learn (dữ liệu nhỏ, trong bộ nhớ) và Apache Spark (dữ liệu lớn, phân tán).

Các file và lệnh thực thi:

- File: test/lab5_test.py

Mô tả: Kiểm tra cơ bản (dữ liệu đơn giản) và sử dụng dữ liệu Hugging Face.

Lệnh Thực thi: python -m test.lab5_test

- File: test/lab5_spark_sentiment_analysis.py

Mô tả: Baseline PySpark (Logistic Regression) trên dữ liệu sentiments.csv.

Lệnh Thực thi: python -m test.lab5_spark_sentiment_analysis

- File: test/lab5_spark_pipeline_nb.py, test/lab5_spark_pipeline_word2vec.py, test/lab5_spark_pipeline_improved_v2.py,test/lab5_spark_pipeline_gpt.py

Mô tả: Cải tiển bằng các mô hình (LR, NB, GBT)

Lệnh Thực thi: python -m test.lab5_spark_pipeline_nb , ...

## II. Phân Tích Kết Quả Ban Đầu

Ban đầu với lab5_test, em dùng bộ dữ liệu giả định đơn giản như trong task 1 thì được kết quả này:

KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Dữ liệu Giả định - 10 mẫu)

Accuracy: 0.5000

Precision: 0.5000

Recall: 1.0000

F1_score: 0.6667

Phân tích:
Kết quả này cho thấy mô hình Logistic Regression đã học theo quy tắc "dự đoán mọi thứ là Positive" do tập dữ liệu huấn luyện quá nhỏ (8 mẫu). Recall 1.0000 cho thấy mô hình nhận diện đúng 100% mẫu Positive, nhưng Precision 0.5000 cho thấy nó cũng dự đoán sai mẫu Negative thành Positive. Kết quả này không phản ánh hiệu suất thực tế.

Sau khi áp dụng bộ dữ liệu Hugging Face (1000 mẫu) thì độ chính xác khả quan hơn, chứng tỏ chất lượng và số lượng dữ liệu đóng vai trò quyết định.

KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Dữ liệu Hugging Face - 1000 mẫu)

Accuracy: 0.8750

Precision: 0.8594

Recall: 0.7746

F1_score: 0.8148

### III. Phân Tích Task Nâng Cao (PySpark)

Đối với bộ dữ liệu sentiments.csv lớn hơn (5791 mẫu), em đã thực hiện các thử nghiệm với PySpark:

1. Baseline: Logistic Regression (PySpark)

Mô hình nền tảng sử dụng TF-IDF (10,000 features).

KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG (PySpark LR)

Accuracy: 0.7295

Precision: 0.7688

Recall: 0.8110

F1_score: 0.7266

# Kết quả các mô hình NLP (Task 1)

| Kỹ thuật/Cải tiến              | Mô hình            | Accuracy | Precision | Recall  | F1-score |
|--------------------------------|------------------|---------|-----------|---------|----------|
| Cải tiến Feature (Tốt nhất)    | Logistic Regression | 0.7394 | 0.7867    | 0.8206  | **0.7368** |
| Thay thế Mô hình               | Naive Bayes        | 0.6907 | 0.7522    | 0.7532  | 0.6906 |
| Mô hình Ensemble               | GBTClassifier      | 0.7286 | 0.7117    | 0.9509  | 0.6953 |
| Word2Vec Embeddings            | Word2Vec + LR     | 0.6637 | 0.6720    | 0.9019  | 0.6212 |