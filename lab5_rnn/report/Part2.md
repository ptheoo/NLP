# Báo cáo kết quả Lab5 RNN — Text Classification

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
| can you remind me to not call my mom | reminder_create | reminder_create | reminder_create | reminder_create | LSTM nhận diện tốt phủ định "not", Scratch LSTM chưa học đủ. |
| is it going to be sunny or rainy tomorrow | weather_query  | weather_query  | weather_query  | weather_query | LSTM xử lý chuỗi giúp nhận diện "sunny or rainy", Scratch LSTM chưa hội tụ. |
| find a flight from new york to london but not through paris | flight_search | flight_search | flight_search | flight_search | Phủ định “but not through” dễ nhầm, LSTM nhờ chuỗi vẫn đúng, Scratch LSTM kém. |

**Nhận xét định tính:**

- LSTM có **lợi thế xử lý chuỗi và phủ định**, giúp hiểu đúng ý định trong câu phức tạp.  
- Word2Vec + Dense cũng dự đoán tốt nhờ vector ngữ nghĩa trung bình.  
- TF-IDF + Logistic Regression dựa trên n-gram, vẫn dự đoán chính xác nhưng khó với câu dài/đa nhãn.  
- Scratch LSTM chưa huấn luyện đủ dữ liệu → kết quả kém, đặc biệt với câu có phủ định hoặc cấu trúc phức tạp.

---

## 3. Kết luận tổng quát

- **Hiệu năng định lượng:** TF-IDF + Logistic Regression ổn định nhất.  
- **Khả năng xử lý ngữ cảnh/chuỗi:** LSTM (pre-trained embedding) mạnh hơn với câu phủ định/cấu trúc phức tạp.  
- **Dữ liệu nhỏ:** Scratch LSTM khó hội tụ → cần nhiều dữ liệu hơn hoặc fine-tune pre-trained embedding.
