# Báo Cáo Lab 4

## I. Tổng quan và Hướng dẫn Thực thi (Task 1, 2, 3)


Dự án này triển khai hệ thống phân loại văn bản nhị phân trên hai framework: Scikit-learn (dữ liệu nhỏ, trong bộ nhớ) và Apache Spark (dữ liệu lớn, phân tán).
**Các Bước Triển Khai (Implementation Steps):**

Quá trình triển khai tập trung vào việc xây dựng Pipeline Phân loại Văn bản (Text Classification Pipeline), bao gồm 4 giai đoạn chính:

1. Thu thập và Chuẩn hóa Dữ liệu:

- Tải dữ liệu (từ giả định, Hugging Face, hoặc sentiments.csv).

- Sử dụng hàm của PySpark/Scikit-learn để làm sạch cơ bản (xóa các hàng thiếu nhãn/văn bản) và chuẩn hóa nhãn (ví dụ: chuyển -1/1 thành 0/1).

- Chia dữ liệu thành tập huấn luyện (Train) và tập kiểm thử (Test) (thường là tỷ lệ 80/20).

2. Tiền xử lý Văn bản (Vector hóa): Giai đoạn này biến đổi văn bản thô thành định dạng số mà mô hình có thể hiểu được.

- Tokenizer: Tách câu thành từng từ (token).

- StopWordsRemover: Loại bỏ các từ vô nghĩa, phổ biến (như "is", "the", "a").

- HashingTF/TF-IDF:

HashingTF ánh xạ các từ đã được làm sạch thành một vector đặc trưng thưa (sparse vector) có kích thước cố định.

IDF (Inverse Document Frequency) tính trọng số cho các vector này, giảm bớt tầm quan trọng của các từ xuất hiện quá thường xuyên trong toàn bộ corpus. Vector đầu ra chính là đầu vào cho mô hình.

3. Huấn luyện Mô hình:

- PySpark: Các bước tiền xử lý và mô hình được lắp ráp vào một Pipeline duy nhất. Lệnh pipeline.fit(trainingData) sẽ tự động thực hiện tất cả các bước (từ Tokenizer đến TF-IDF) trên dữ liệu huấn luyện, sau đó huấn luyện mô hình phân loại (Logistic Regression, Naive Bayes, hoặc GBT) trên các vector đặc trưng cuối cùng.

- Scikit-learn: Lệnh vectorizer.fit_transform() và sau đó model.fit() thực hiện tương tự.

4. Đánh giá Hiệu suất:

- Mô hình đã huấn luyện được áp dụng cho tập kiểm thử (model.transform(testData)).

- Sử dụng MulticlassClassificationEvaluator của PySpark để tính toán Accuracy và F1-Score.

- Các chỉ số Precision và Recall được tính toán thủ công từ ma trận nhầm lẫn ($\text{Confusion Matrix}$) để đánh giá chi tiết khả năng dự đoán đúng Positive (Recall) và độ tin cậy của các dự đoán Positive đó (Precision).

**Các file và lệnh thực thi:**

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

## III. Phân Tích Task Nâng Cao (PySpark)

Đối với bộ dữ liệu sentiments.csv lớn hơn (5791 mẫu), em đã thực hiện các thử nghiệm với PySpark:


### 1. Baseline: Logistic Regression (PySpark)

Mô hình nền tảng sử dụng TF-IDF (10,000 features).

KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG (PySpark LR)

Accuracy: 0.7295

Precision: 0.7688

Recall: 0.8110

F1_score: 0.7266

### 2. Kết quả các mô hình NLP (Task 4)

| Kỹ thuật/Cải tiến              | Mô hình            | Accuracy | Precision | Recall  | F1-score |
|--------------------------------|------------------|---------|-----------|---------|----------|
| Cải tiến Feature (Tốt nhất)    | Logistic Regression | 0.7394 | 0.7867    | 0.8206  | **0.7368** |
| Thay thế Mô hình               | Naive Bayes        | 0.6907 | 0.7522    | 0.7532  | 0.6906 |
| Mô hình Ensemble               | GBTClassifier      | 0.7286 | 0.7117    | 0.9509  | 0.6953 |
| Word2Vec Embeddings            | Word2Vec + LR     | 0.6637 | 0.6720    | 0.9019  | 0.6212 |

**Phân tích:**

+ Cải tiến Hiệu quả nhất: Kỹ thuật giảm số chiều $\text{TF-IDF}$ trong mô hình $\text{LogisticRegression}$ đã mang lại $\mathbf{F1-score}$ cao nhất là $\mathbf{0.7368}$. 
Khi số lượng chiều quá lớn, các từ rất hiếm (như lỗi chính tả, ID, hoặc ký hiệu không chuẩn) cũng được coi là đặc trưng. Những đặc trưng nhiễu này làm mô hình bị quá khớp (overfit) với tập huấn luyện và không tổng quát hóa tốt. Bằng cách giảm chiều, chúng ta buộc mô hình chỉ tập trung vào các đặc trưng có ảnh hưởng và xuất hiện đủ thường xuyên, từ đó cải thiện $\text{Precision}$ và tăng khả năng tổng quát hóa.

Tầm quan trọng: Trong xử lý văn bản (đặc biệt là dữ liệu mạng xã hội như tweets), Làm sạch và Giảm chiều là bước tối quan trọng để chống lại "curse of dimensionality" (lời nguyền chiều dữ liệu) và đảm bảo mô hình học được các quy luật thực tế chứ không phải là nhiễu.

+ Naive Bayes: Hiệu suất thấp hơn $\text{Baseline}$ ($0.6906$ vs $0.7266$) vì mô hình này giả định tính độc lập của các từ, không phù hợp với mối quan hệ ngữ nghĩa trong các tweet tài chính.

+ Word2Vec/GBT: Cả hai mô hình này cho $\text{F1-score}$ thấp hơn $\text{Baseline}$.

+ $\text{GBT}$ cho $\text{Recall}$ rất cao ($0.9509$) nhưng $\text{Precision}$ thấp, cho thấy mô hình quá khớp (overfit) với dữ liệu $\text{TF-IDF}$ thưa thớt, tạo ra quá nhiều False Positives.

+ $\text{Word2Vec}$ cho kết quả kém nhất ($0.6212$) vì mô hình nhúng được huấn luyện trên tập dữ liệu nội bộ quá nhỏ không thể học được ngữ nghĩa từ vựng hiệu quả.