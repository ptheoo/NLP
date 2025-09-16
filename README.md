# NLP — Báo cáo (Lab 1 & Lab 2)

> Đây là báo cáo tóm tắt quá trình đã thực hiện trong **Lab 1** và **Lab 2**: xây dựng tokenizer, loader cho dataset UD_English-EWT, triển khai CountVectorizer, và chạy thử nghiệm.

---

## 1. Mục tiêu

### Lab 1
- Tạo interface `Tokenizer` (chuẩn hóa thiết kế).
- Cài đặt:
  - `SimpleTokenizer`: tách token theo khoảng trắng.
  - `RegexTokenizer`: tách token bằng regex `\w+|[^\w\s]`.
- Viết `load_raw_text_data` để load văn bản từ dataset UD (CoNLL-U).
- Viết script test cho tokenizer trên câu mẫu + dữ liệu thật.

### Lab 2
- Tạo interface `Vectorizer`.
- Cài đặt `CountVectorizer`:
  - `fit(corpus)`: sinh vocabulary.
  - `transform(corpus)`: tạo document-term matrix.
- Test trên corpus nhỏ + một sample từ UD dataset.

---

## 2. Cấu trúc thư mục
```
NLP/
├── src/
│ ├── core/
│ │ ├── interfaces.py # Interface: Tokenizer, Vectorizer
| | └──dataset_loader.py
│ ├── preprocessing/
│ │ ├── simple_tokenizer.py # Tokenizer đơn giản (split)
│ │ └── regex_tokenizer.py # Tokenizer dùng regex
│ ├── core/dataset_loaders.py # Hàm load dữ liệu từ UD .conllu
│ └── representations/count_vectorizer.py # CountVectorizer
├── test/
│ ├── lab1_test1.py # Test SimpleTokenizer cơ bản
│ ├── lab1_test2.py # So sánh SimpleTokenizer vs RegexTokenizer
│ ├── lab1_test3.py # Test tokenizer trên UD dataset
│ └── lab2_test.py # Test CountVectorizer
├── UD_English-EWT/ # Dữ liệu (CoNLL-U)
│ └── en_ewt-ud-train.conllu
└── README.md
```

---

## 3. Chuẩn bị môi trường

Tạo và kích hoạt môi trường ảo (Windows):

```bash
python -m venv venv
venv\Scripts\activate

```
Cài dependencies:
```bash
pip install -r requirements.txt
```
## 4. Cách chạy

Chạy test cho từng phần:
```bash
python -m test.lab1_test1
python -m test.lab1_test2
python -m test.lab1_test3
python -m test.lab2_test
```
## 5. Kết quả & Phân tích
### Lab 1. Task 1 — SimpleTokenizer cơ bản
```bash
python -m test.lab1_test1
```
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']


👉 Tokenizer đã tách từ và dấu câu riêng biệt. 

### Lab 1. Task 2 — So sánh SimpleTokenizer vs RegexTokenizer
```
Sentence: Hello, world! This is a test.
SimpleTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer:   ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Sentence: NLP is fascinating... isn't it?
SimpleTokenizer: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer:   ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Sentence: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer:   ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

👉 Nhận xét:

Với câu đơn giản, 2 tokenizer cho kết quả giống nhau.

Với câu chứa dấu nháy ('), RegexTokenizer chi tiết hơn: "isn't" → ['isn', "'", 't'], "let's" → ['let', "'", 's'].

Điều này phù hợp cho xử lý NLP chuyên sâu (POS tagging, lemmatization).

### Lab 1. Task 3 — UD Dataset Sample
```
--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: From the AP comes this story : President Bush ...
SimpleTokenizer Output (first 20 tokens): ['from', 'the', 'ap', 'comes', 'this', 'story', ':', 'president', 'bush', 'on', 'tuesday', 'nominated', 'two', 'individuals', 'to', 'replace', 'retiring', 'jurists', 'on', 'federal']
RegexTokenizer Output (first 20 tokens):  ['from', 'the', 'ap', 'comes', 'this', 'story', ':', 'president', 'bush', 'on', 'tuesday', 'nominated', 'two', 'individuals', 'to', 'replace', 'retiring', 'jurists', 'on', 'federal']
```

👉 Nhận xét:

Trên dữ liệu thực, 2 tokenizer cho kết quả tương đồng ở 20 token đầu.

RegexTokenizer vẫn có lợi thế ở các câu chứa ký tự đặc biệt phức tạp.

### Lab 2 — CountVectorizer
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

👉 Nhận xét:

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
👉 Nhận xét:

Vocabulary lớn, phản ánh tính đa dạng của dữ liệu thật.

Ma trận sparse, đặc trưng cho văn bản lớn.

CountVectorizer đã biểu diễn thành công corpus thành vector space.

## 6. Kết luận

Lab 1: Đã xây dựng 2 tokenizer hoạt động ổn, RegexTokenizer chi tiết hơn.

Lab 2: CountVectorizer triển khai thành công, áp dụng tốt cả trên corpus nhỏ và dataset thật.

Hệ thống hiện đã có pipeline cơ bản: tokenizer → vectorizer → matrix representation.

Tiếp theo có thể mở rộng với:

TF-IDF Vectorizer.

Loại bỏ stopword.

Word embeddings (Word2Vec, BERT).