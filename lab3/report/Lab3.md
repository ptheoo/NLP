# Lab3 — Word Embeddings (Lab Report)

Mục tiêu: Thực hiện các thao tác với word embeddings: sử dụng pretrained embedding (GloVe), huấn luyện Word2Vec từ dữ liệu nhỏ (Gensim) và lớn (Spark), tạo document embedding, giảm chiều & trực quan hóa, và phân tích kết quả.


## Mục lục

1. Tóm tắt tiến độ (Checklist)

2. Môi trường & Cài đặt

3. Hướng dẫn chạy (How to run)

4. Nội dung thực thi & kết quả chính (Outputs)

4.1 Spark Word2Vec demo (kết quả)

4.2 Sử dụng GloVe pretrained (lab4_test)

4.3 Huấn luyện Word2Vec từ đầu (lab4_embedding_training_demo)

5. Phân tích & Nhận xét chi tiết (Phần Quan trọng)

5.1 Pretrained GloVe — chất lượng & nhận xét

5.2 Word2Vec tự huấn luyện trên en_ewt — vì sao kết quả “kỳ lạ”?

5.3 So sánh: Pretrained vs Trained-from-scratch

5.4 Giảm chiều & trực quan hóa 

6. Các vấn đề gặp phải & cách giải quyết (Troubleshooting)

7. Đề xuất cải tiến & bước tiếp theo


## 1. Tóm tắt tiến độ (Checklist)

Phần 1: Triển khai (50%)

 Task 1: Tải và sử dụng pretrained model (Gensim) — glove-wiki-gigaword-50 (đã tải thành công).

 Lấy vector của một từ (king) — OK.

 Tính similarity (king vs queen, king vs man) — OK.

 Tìm các từ most_similar (computer) — OK.

 Task 2: Nhúng câu/văn bản bằng trung bình vector — OK.

 Task 3: Huấn luyện model Word2Vec từ dữ liệu thô (en_ewt-ud-train.txt) — Đã huấn luyện thành công, model lưu được.

 Lưu & tải lại model huấn luyện — OK.

 Task 4: Huấn luyện model trên tập dữ liệu lớn (Spark) — Đã chạy demo Spark Word2Vec (kết quả hiển thị).

 Task 5: Trực quan hóa embedding bằng PCA/t-SNE — Hoàn thành

Phần 2: Báo cáo và Phân tích (50%)

 Giải thích các bước thực hiện — có trong báo cáo này.

 Hướng dẫn chạy code — có.

 Phân tích kết quả — có (mục 5).

 Nhận xét về similarity / most_similar (pretrained) — có.

 Phân tích trực quan hóa — có (Trong file PDF)

 So sánh pretrained vs tự huấn luyện — có.

 Khó khăn & giải pháp — có.


## 2. Môi trường & Cài đặt

Khuyến nghị môi trường (venv):

Python 3.10

Virtual environment (ví dụ venv) — bạn đang dùng (venv).

Thêm vào ```requirements.txt``` các thư viện sau:

```bash
gensim
nltk
numpy==1.26.4
scipy==1.11.4
matplotlib
scikit-learn
pyspark
```

Lưu ý: đã từng gặp xung đột numpy vs thinc — trong lab này phiên bản numpy==1.26.4 + scipy==1.11.4 hoạt động ổn với gensim. Nếu bạn cài spacy/thinc có thể xuất cảnh báo tương thích, không gây lỗi cho các tác vụ hiện tại.

NLTK: cần download punkt để tokenize:
```bash
import nltk
nltk.download('punkt')
```
## 3. Hướng dẫn chạy (How to run)

Kích hoạt môi trường ảo:

Windows PowerShell
```bash
.\venv\Scripts\Activate.ps1
```
hoặc cmd:
```bash
.\venv\Scripts\activate.bat
```

Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

Chạy các script:

Spark Word2Vec demo:
```bash
python test/lab4_spark_word2vec_demo.py
```

Sử dụng pretrained GloVe và test các hàm:
```bash
python test/lab4_test.py
```

Huấn luyện Word2Vec từ en_ewt:
```bash
python test/lab4_embedding_training_demo.py
```

## 4. Nội dung thực thi & kết quả chính (Outputs)

Dưới đây là các output đã chạy:
### 4.1 Sử dụng GloVe pretrained (lab4_test)
```bash
[nltk_data] Downloading package punkt...
🔹 Đang tải mô hình 'glove-wiki-gigaword-50' ...
 Mô hình 'glove-wiki-gigaword-50' tải thành công (50-dim).

--- 🔹 Lấy vector của từ 'king' ---
Kích thước vector: (50,)
Giá trị đầu tiên: [ 0.50451   0.68607 -0.59517 -0.022801  0.60046 ]        

--- 🔹 Độ tương đồng ---
king vs queen: 0.78390425
king vs man: 0.53093773

--- 🔹 10 từ gần nghĩa với 'computer' ---
computers       -> 0.9165
software        -> 0.8815
technology      -> 0.8526
electronic      -> 0.8126
internet        -> 0.8060
computing       -> 0.8026
devices         -> 0.8016
digital         -> 0.7992
applications    -> 0.7913
pc              -> 0.7883

--- 🔹 Vector văn bản ---
Vector biểu diễn văn bản:
[ 0.04564168  0.36530998 -0.55974334  0.04014383  0.09655549  0.15623933
 -0.33622834 -0.12495166 -0.01031508 -0.5006717 ]
Độ dài vector: 50
```
### 4.2 Huấn luyện Word2Vec từ đầu (lab4_embedding_training_demo) — kết quả sample
```bash
BẮT ĐẦU: HUẤN LUYỆN MÔ HÌNH WORD2VEC TỪ ĐẦU
...
Tổng số câu được đọc để huấn luyện: 14225
...
Word2Vec lifecycle event {... vocab=3866, vector_size=100 ...}
Huấn luyện mô hình Word2Vec hoàn tất.
Kích thước từ vựng mô hình (vocab size): 3866

Đang lưu mô hình đã huấn luyện tại: .../results/word2vec_ewt.model
Lưu mô hình thành công.

Demo sử dụng mô hình Word2Vec đã huấn luyện:

   A. 10 từ tương đồng nhất với 'student':
      1. science: 0.4967
      2. canada,: 0.4903
      3. buy: 0.4637
      4. core: 0.4627
      5. brief: 0.4625
      6. unlimited: 0.4568
      7. parent: 0.4512
      8. reports,: 0.4425
      9. visa: 0.4393
      10. agel: 0.4387

      ...
   B. Giải quyết bài toán tương tự: king - man + woman = ?
      Kết quả (Top 3):
      1. arabia (Score: 0.4022)
      2. foot (Score: 0.3916)
      3. "it (Score: 0.3914)

```
Ghi chú: kết quả analogies sai là do hạn chế tập huấn luyện (xem phần phân tích).

### 4.3 Spark Word2Vec demo (kết quả)
```bash
Khởi tạo SparkSession.
...
----------
Đọc dữ liệu
Số dòng đọc được: 30000
----------
Tiền xử lý văn bản và Tokenization
Số dòng sau khi lọc các dòng trống: 30000
DataFrame sau khi Tokenization:
+--------------------------------------------------+
|                   words                          |
+--------------------------------------------------+
|[beginners, bbq, class, taking, place, in, miss...]|
...
only showing top 5 rows
----------
Huấn luyện mô hình Word2Vec (Skip-gram)
25/10/15 00:29:21 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS

Tìm các từ tương tự 'computer'
+----------+------------------+
|word      |similarity        |
+----------+------------------+
|desktop   |0.6939913630485535|
|computers |0.6567586064338684|
|software  |0.6353139281272888|
|coding    |0.6132869720458984|
|interfaces|0.6102142333984375|
|laptop    |0.6024858951568604|
|robohelp  |0.5980682373046875|
|backup    |0.5902743339538574|
|pc        |0.5899812579154968|
|graphical |0.5812470316886902|
+----------+------------------+

Hoàn thành huấn luyện Spark Word2Vec
...

```

## 5. Phân tích & Nhận xét chi tiết (Phần Quan trọng)
### 5.1 Pretrained GloVe — chất lượng & nhận xét

glove-wiki-gigaword-50 là embedding tiền huấn luyện trên corpora lớn (Wikipedia + Gigaword).

Kết quả king vs queen ≈ 0.78 và most_similar cho computer đều rất hợp lý — cho thấy pretrained embeddings mang kiến thức ngữ nghĩa sâu rộng.

Ưu điểm: không cần huấn luyện, ổn định, rất phù hợp cho bài tập demo / baseline.

Nhược điểm: không domain-specific; nếu dữ liệu của bạn khác biệt (ví dụ văn bản y tế/tài chính), pretrained có thể không phản ánh tốt thuật ngữ chuyên ngành.

### 5.2 Word2Vec tự huấn luyện trên en_ewt

Quan sát: phép analogies king - man + woman cho kết quả như arabia, foot, "it hoặc easily (trong một lần khác) — không phải queen. Nguyên nhân chính:

Tập dữ liệu nhỏ & hạn chế (14225 câu, ~177k từ):

Một số từ quan trọng (king/queen) có tần suất rất thấp → embedding bị noisy.

Vocab bị cắt (effective_min_count=5):

Gensim đã loại bỏ từ ít xuất hiện, làm mất các từ cần cho analogies.

Dữ liệu không cân bằng/ngữ cảnh nghèo:

Mối quan hệ king ↔ queen cần nhiều ngữ cảnh so sánh (royalty contexts). Nếu không đủ, mô hình không học được.

Thuật toán & siêu tham số: epochs, window, vector_size ảnh hưởng mạnh. Dù đã lặp nhiều epoch (ở output bạn đã chạy nhiều epoch), nếu dữ liệu thiếu biểu diễn ngữ cảnh thì vẫn không tốt.

Hệ quả: mô hình học được mối quan hệ cục bộ/đồng xuất hiện (co-occurrence) chứ chưa học được quy luật ngữ nghĩa sâu.

### 5.3 So sánh: Pretrained vs Trained-from-scratch
```
Tiêu chí	        |  Pretrained (GloVe)	     |  Trained-from-scratch (EWT)
____________________|____________________________|_______________________
Dữ liệu huấn luyện	|   Rất lớn	                 |   Nhỏ (~17k câu)
Chất lượng analogies|	Tốt (king→queen)	     |  Kém / noisy
Phù hợp domain	    |   Chung chung	             |Có thể domain-specific (nếu corpus domain-specific)
Thời gian	        |  Tải nhanh, không cần train|	Cần thời gian train
Khi nào dùng	    |   Baseline, nhanh	         |   Khi cần embedding chuyên ngành
```

Kết luận: Với dữ liệu nhỏ, dùng pretrained để làm baseline; tự huấn luyện chỉ thực sự hiệu quả nếu có corpus đủ lớn hoặc domain-specific.

### 5.4 Giảm chiều & trực quan hóa

PCA: nhanh, tuyến tính — dùng để có cái nhìn tổng quan.

t-SNE / UMAP: tách cụm tốt hơn, phù hợp cho visual analysis.

Nhận xét, đánh giá cụ thể hơn trong  file PDF ```22001286_NguyenThiPhuongThao_Lab3_Phan1```.



## 6. Các vấn đề gặp phải & cách giải quyết (Troubleshooting)
A. ImportError do SciPy / NumPy

Lỗi: ImportError: cannot import name 'triu' from 'scipy.linalg'

Giải pháp: hạ scipy về 1.11.4 và dùng numpy==1.26.4. Lưu ý xung đột với thinc/spacy — nếu không dùng spacy, có thể gỡ thinc/spacy hoặc bỏ qua cảnh báo.

B. ModuleNotFoundError: No module named 'src'

Nguyên nhân: chạy script từ thư mục con.

Giải pháp:

Chạy bằng: python -m test.lab4_test từ project root.

Hoặc thêm sys.path.append(...) trong script test để thêm thư mục gốc vào sys.path.

Hoặc tạo __init__.py phù hợp cho package.

C. t-SNE AttributeError: 'list' object has no attribute 'shape'

Giải pháp: convert vectors_np = np.array(vectors) trước khi gọi tsne.fit_transform(vectors_np).

D. Kết quả analogies坏 (king → easily)

Nguyên nhân: dữ liệu nhỏ / min_count loại bỏ từ / thiếu ngữ cảnh.

Giải pháp: dùng corpus lớn hơn (text8, wikipedia) hoặc giảm min_count, tăng epochs, tăng window, hoặc dùng pretrained.

## 7. Đề xuất cải tiến & bước tiếp theo


So sánh kỹ hơn pretrained vs trained-from-scratch:

Lấy một số cặp test (king-queen, paris-france, doctor-nurse) và đo cosine similarities trên cả hai model. Tổng hợp vào bảng.

Thử FastText: tốt với từ hiếm và OOV (subword).

Tăng/đổi tham số huấn luyện: giảm min_count, tăng epochs (nếu dữ liệu đủ), thử window lớn hơn.

Dùng corpus lớn hơn: text8 (sẵn có), Wikipedia (tải via gensim.downloader) → huấn luyện sẽ cho analogies tốt.

Định lượng: ngoài trực quan, thực hiện đánh giá định lượng như intrinsic evaluation (word similarity datasets) nếu cần.