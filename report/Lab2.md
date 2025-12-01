# Lab2
## 1. Mục tiêu
- Tạo interface `Vectorizer`.
- Cài đặt `CountVectorizer`:
  - `fit(corpus)`: sinh vocabulary.
  - `transform(corpus)`: tạo document-term matrix.
- Test trên corpus nhỏ + một sample từ UD dataset.



## 2. Chuẩn bị môi trường

Tạo và kích hoạt môi trường ảo (Windows):

```bash
python -m venv venv
venv\Scripts\activate

```
Cài dependencies:
```bash
pip install -r requirements.txt
```
---
## 3. Cách chạy

Chạy test cho từng phần:
```bash


python -m test.lab2_test
```
---

## 4. Kết quả và phân tích
```
Corpus:

I love NLP.
I love programming.
NLP is a subfield of AI.


Vocabulary:

.: 0
a: 1
ai: 2
i: 3
is: 4
love: 5
nlp: 6
of: 7
programming: 8
subfield: 9


Document-Term Matrix:

[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

 Nhận xét:

Vector biểu diễn tần suất token.

Ví dụ: câu I love NLP. có i=1, love=1, nlp=1.

UD Dataset Sample (rút gọn):
```
Vocabulary (trích):

*: 0
,: 1
-: 2
.: 3
15: 4
...
washington: 70
wheel: 71
with: 72
year: 73
years: 74


Document-Term Matrix (3 dòng đầu, rút gọn):

[0, 0, 1, 2, 1, 0, 1, 1, ...]
[3, 1, 1, 3, 1, 0, 0, 2, ...]
[0, 1, 3, 2, 0, 1, 0, 1, ...]

```
Nhận xét:

Vocabulary lớn, phản ánh tính đa dạng của dữ liệu thật.

Ma trận sparse, đặc trưng cho văn bản lớn.

CountVectorizer đã biểu diễn thành công corpus thành vector space.

---
## 5.Kết luận

CountVectorizer đã triển khai thành công, áp dụng tốt cả trên corpus nhỏ và dataset thật.

Hệ thống hiện đã có pipeline cơ bản: tokenizer → vectorizer → matrix representation.

Tiếp theo có thể mở rộng với:

TF-IDF Vectorizer.

Loại bỏ stopword.

Word embeddings (Word2Vec, BERT).