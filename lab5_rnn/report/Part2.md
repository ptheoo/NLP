# Report Part 2: RNN — Text Classification

## 1. Bảng tổng hợp kết quả định lượng (Test Set)

| Pipeline | F1-score (Macro) | Test Loss |
|----------|-----------------|-----------|
| TF-IDF + Logistic Regression | 0.83 | N/A |
| Word2Vec (Avg) + Dense | ~0.71 | 0.913 |
| Embedding (Pre-trained) + LSTM | ~0.43 | 1.8825 |
| Embedding (Scratch) + LSTM | ~0.24 | 2.6339 |

**Nhận xét:**

- **TF-IDF + Logistic Regression** là baseline mạnh nhất về F1-score macro, đặc biệt với các lớp ít mẫu.  
- **Word2Vec (Avg) + Dense** cải thiện khả năng biểu diễn ngữ nghĩa nhưng vẫn thua baseline. Test loss thấp (0.91) cho thấy mô hình ổn định.  
- **Embedding + LSTM** (cả Pre-trained và Scratch) có F1-score thấp, chứng tỏ việc huấn luyện LSTM trên tập dữ liệu nhỏ chưa đủ để vượt trội.  
- Test loss cao ở LSTM do mô hình phức tạp và khó hội tụ trên tập nhỏ.

---

## 2. Phân tích định tính

Ví dụ các câu kiểm tra “khó”:

1. **“can you remind me to not call my mom”** → nhãn thật: `reminder_create`  
2. **“is it going to be sunny or rainy tomorrow”** → nhãn thật: `weather_query`  
3. **“find a flight from new york to london but not through paris”** → nhãn thật: `flight_search`  

| Câu | TF-IDF + LR | Word2Vec + Dense | Pretrained LSTM | Scratch LSTM | Nhận xét |
|-----|-------------|-----------------|----------------|--------------|----------|
| can you remind me to not call my mom | reminder_create | transport_ticket  | lists_createoradd | reminder_create | Câu này có phủ định “not call”, TF-IDF + LR dự đoán đúng nhờ từ khóa “remind”, Word2Vec + Dense bị nhầm. Pretrained LSTM và Scratch LSTM khác nhau: Scratch LSTM dự đoán đúng nhờ khả năng hiểu thứ tự từ và phủ định, trong khi Pretrained LSTM dự đoán sai, có thể do embedding không phù hợp với từ “not”. Khả năng xử lý chuỗi giúp Scratch LSTM nhận ra mối quan hệ phủ định. |
| is it going to be sunny or rainy tomorrow | weather_query  | weather_query  | music_query    | cooking_recipe  | Câu này là dự đoán thời tiết với từ khóa “sunny” và “rainy”. TF-IDF và Word2Vec dự đoán chính xác nhờ học từ khóa. Hai mô hình LSTM lại dự đoán sai, có thể vì chúng chú trọng vào cấu trúc câu và embedding chưa bắt kịp ngữ cảnh thời tiết, dẫn đến dự đoán lệch. Khả năng xử lý chuỗi không chắc giúp LSTM trong trường hợp này nếu từ khóa quan trọng bị embedding yếu.. |
| find a flight from new york to london but not through paris | flight_search | transport_query   | flight_search | flight_search | Đây là câu phức tạp, có phủ định “not through paris” và nhiều địa điểm. Cả hai LSTM đều dự đoán đúng, trong khi TF-IDF dự đoán đúng nhờ từ khóa “flight”, Word2Vec hơi lệch. LSTM thể hiện ưu thế nhờ khả năng xử lý thứ tự từ và ngữ cảnh dài, hiểu rằng người dùng muốn bay từ New York → London nhưng không qua Paris.. |

**Nhận xét định tính:**

- LSTM có **lợi thế xử lý chuỗi và phủ định**, giúp hiểu đúng ý định trong câu phức tạp.  
- Word2Vec + Dense cũng dự đoán tốt nhờ vector ngữ nghĩa trung bình.  
- TF-IDF + Logistic Regression dựa trên n-gram, vẫn dự đoán chính xác nhưng khó với câu dài/đa nhãn.  
- Scratch LSTM chưa huấn luyện đủ dữ liệu → kết quả kém, đặc biệt với câu có phủ định hoặc cấu trúc phức tạp.

**Nhận xét chung**

- Ưu điểm của LSTM:

Có khả năng xử lý chuỗi và thứ tự từ.

Nhận diện phủ định, mệnh đề phức tạp, và mối quan hệ từ xa trong câu.

Hiệu quả hơn trong các câu dài, phức tạp hoặc có cấu trúc phụ thuộc xa.

- Nhược điểm của LSTM:

Cần dữ liệu huấn luyện nhiều và embedding phù hợp để dự đoán chính xác.

Đôi khi nhạy cảm với từ khóa nổi bật nếu embedding chưa tốt, dẫn đến dự đoán sai (như ví dụ thời tiết ở trên).

- So sánh với TF-IDF / Word2Vec + Dense:

TF-IDF + LR mạnh ở các câu ngắn, có từ khóa đặc trưng.

Word2Vec + Dense trung bình, đôi khi dự đoán sai nếu mối quan hệ từ xa quan trọng.

LSTM vượt trội khi câu phức tạp, phủ định hoặc nhiều nhánh ý nghĩa.

---

## 3. Nhận xét chung về ưu và nhược điểm từng phương pháp

| Mô hình | Ưu điểm | Nhược điểm |
|---------|---------|------------|
| TF-IDF + Logistic Regression | - Đơn giản, nhanh. <br> - F1 macro cao, ổn định trên lớp ít mẫu. | - Không nắm ngữ cảnh dài hoặc phủ định. <br> - Khó mở rộng với dữ liệu phức tạp. |
| Word2Vec (Avg) + Dense | - Giữ được ngữ nghĩa từ các từ trong câu. <br> - Dự đoán khá ổn với từ khóa. | - Trung bình vector → mất thứ tự từ, khó xử lý phủ định. |
| Embedding (Pre-trained) + LSTM | - Xử lý chuỗi tốt, nhận diện phủ định, cấu trúc phức tạp. | - Cần nhiều dữ liệu để huấn luyện. <br> - Test F1 macro thấp với tập nhỏ. |
| Embedding (Scratch) + LSTM | - Có tiềm năng học chuỗi và ngữ cảnh. | - Dữ liệu nhỏ → khó hội tụ, F1 thấp, loss cao. |

**Kết luận:**  
- Với dữ liệu nhỏ, TF-IDF + Logistic Regression vẫn là lựa chọn baseline mạnh mẽ.  
- LSTM (đặc biệt embedding pre-trained) thể hiện sức mạnh xử lý chuỗi, nhưng cần **dữ liệu nhiều hơn hoặc fine-tune embedding** để vượt trội.  
- Word2Vec + Dense là bước trung gian tốt giữa n-gram và LSTM.
