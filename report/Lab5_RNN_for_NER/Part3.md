# Report part 3: POS Tagging

## 1. Thông tin dữ liệu và từ điển

- Bộ dữ liệu: **Universal Dependencies (UD_English-EWT)**  
- Kích thước từ điển: `Vocabulary size = 20201`  
- Số nhãn POS: `Number of tags = 18`  

---

## 2. Huấn luyện mô hình RNN

- Mô hình: **SimpleRNNForTokenClassification**  
- Tham số:  
  - Embed_dim = 100  
  - Hidden_dim = 128  
  - Batch size = 32  
  - Epochs = 5  

### 2.1 Quá trình huấn luyện

| Epoch | Loss     | Train Acc | Dev Acc |
|-------|----------|-----------|---------|
| 1     | 1.1665   | 0.7664    | 0.7469  |
| 2     | 0.6520   | 0.8330    | 0.8064  |
| 3     | 0.4906   | 0.8724    | 0.8335  |
| 4     | 0.3902   | 0.8972    | 0.8466  |
| 5     | 0.3195   | 0.9166    | 0.8538  |

**Nhận xét:**  

- Độ chính xác trên tập dev tăng dần qua các epoch → mô hình học tốt.  
- Mô hình tốt nhất dựa trên **Dev Accuracy**: **epoch 5 (0.8538)**.  
- Loss giảm dần, chứng tỏ mô hình hội tụ ổn định.  

---

## 3. Đánh giá mô hình trên tập dev

- **Accuracy cuối cùng trên tập dev**: **0.8538**  
- Mô hình RNN học mối quan hệ giữa các từ trong câu → dự đoán nhãn POS chính xác hơn các phương pháp không xử lý chuỗi.  

---

## 4. Dự đoán câu mới

### Ví dụ 1
- Câu: `I love NLP`  
- Dự đoán nhãn POS: 
```bash
[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN')]
```

### Ví dụ 2:
- Câu: `The quick brown fox jumps over the lazy dog`  
- Dự đoán nhãn POS: 
```bash
[('The', 'DET'), ('quick', 'ADJ'), ('brown', 'PROPN'), ('fox', 'NOUN'),
('jumps', 'VERB'), ('over', 'ADP'), ('the', 'DET'), ('lazy', 'ADJ'), ('dog', 'NOUN')]
```


**Giải thích:**  

- `'I'` → PRON (đại từ)  
- `'love'` → VERB (động từ)  
- `'NLP'` → PROPN (danh từ riêng)  
- `'The'` → DET, `'quick'` → ADJ, `'brown'` → PROPN, `'fox'` → NOUN, ...  

Hàm `predict_sentence(sentence)` có thể sử dụng trực tiếp để dự đoán nhãn POS cho bất kỳ câu tiếng Anh nào.  

---

## 5. Kết luận

- Mô hình RNN học được mối quan hệ tuần tự giữa các từ → dự đoán POS chính xác trên câu mới.  
- Accuracy **~85% trên tập dev** là kết quả tốt cho bài toán POS tagging cơ bản.  
- Mô hình có thể mở rộng hoặc thay RNN bằng LSTM/GRU để cải thiện hiệu suất trên các câu phức tạp hơn.
