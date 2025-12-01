## 1. Tổng quan Dự án

Project này là tập hợp các bài tập và báo cáo **Thực hành (Lab)** chuyên sâu về các kỹ thuật và mô hình Học sâu (Deep Learning) trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP). Dự án tập trung vào việc **triển khai các mô hình từ cấp độ cơ bản đến phức tạp** để giải quyết các nhiệm vụ NLP quan trọng.

### Mục tiêu Cốt lõi
* **Triển khai Mô hình Chuỗi:** Áp dụng Mạng Nơ-ron Hồi quy (RNN/LSTM/GRU) để xử lý dữ liệu tuần tự.
* **Các Nhiệm vụ NLP:** Thực hành các bài toán nền tảng như:
    * Phân loại Văn bản (Text Classification).
    * Gắn thẻ Từ loại (Part-of-Speech Tagging - POS Tagging).
    * Nhận dạng Thực thể có Tên (Named Entity Recognition - NER).
* **Kỹ năng Tổ chức:** Rèn luyện khả năng tổ chức mã nguồn, dữ liệu và trình bày kết quả qua các báo cáo học thuật.

### Công nghệ Nền tảng
| Khía cạnh | Công nghệ |
| :--- | :--- |
| **Ngôn ngữ** | Python (3.x) |
| **Frameworks** | PyTorch (dựa trên các file code) |
| **Thư viện chính** | NumPy, Pandas, Scikit-learn, các thư viện NLP tiêu chuẩn. |

---

## 2. Cấu trúc

Cấu trúc dự án được phân chia theo từng Bài Lab, giúp dễ dàng theo dõi tiến trình học tập và kiểm tra kết quả của từng nhiệm vụ cụ thể.
```
.
├── data/                      #  Dữ liệu gốc và đã xử lý (chung)
├── lab1_2/                    #  Thực hành Lab 1 & 2 (NLP Cơ bản)
├── lab3/                      # Thực hành Lab 3
├── lab5_text_classification/  # Thực hành Lab 4: Text classification
├── lab5_rnn/                  # Thực hành Lab 5 chuyên sâu về Mô hình RNN
├── report/                    # Báo cáo tất cả các bài lab
├── venv/                      # Môi trường ảo Python
├── .gitignore
├── README.md
└── requirements.txt
```
---

## 3. Hướng dẫn Cài đặt & Sử dụng

### 3.1. Cài đặt Môi trường
Để đảm bảo môi trường chạy code ổn định và không xung đột với các dự án khác, hãy tạo và kích hoạt môi trường ảo:

1.  **Tạo môi trường ảo:**
    ```bash
    python -m venv venv
    ```
2.  **Kích hoạt môi trường:**
    * **Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        .\venv\Scripts\activate
        ```
3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

### 3.2. Chạy Code Thực hành
Code cho từng nhiệm vụ nằm trong các thư mục `part` của `lab5_rnn/`.

* **Để chạy ví dụ Phân loại Văn bản (Text Classification):**
    ```bash
    cd lab5_rnn/part2/
    python lab5_rnn_text_class_...py
    ```
* **Để chạy ví dụ Gắn thẻ Từ loại (POS Tagging):**
    ```bash
    cd lab5_rnn/part3/
    python lab5_rnn_for_pos_t...py
    ```

### 3.3. Xem Báo cáo
Tất cả báo cáo chi tiết (phương pháp, kết quả, phân tích lỗi) được lưu trữ tại `report/`. Hãy mở các file `.md` (hoặc `.pdf`) tương ứng để xem tài liệu.