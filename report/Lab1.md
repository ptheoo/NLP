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
---


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
## 3. Cách chạy

Chạy test cho từng phần:
```bash
python -m test.lab1_test1
python -m test.lab1_test2
python -m test.lab1_test3

```
## 4. Kết quả & Phân tích
### Lab 1. Task 1 — SimpleTokenizer cơ bản
```bash
python -m test.lab1_test1
```
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']


Tokenizer đã tách từ và dấu câu riêng biệt. 

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

Nhận xét:

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

Nhận xét:

Trên dữ liệu thực, 2 tokenizer cho kết quả tương đồng ở 20 token đầu.

RegexTokenizer vẫn có lợi thế ở các câu chứa ký tự đặc biệt phức tạp.

## 5. Kết luận

Lab 1 đã xây dựng 2 tokenizer hoạt động ổn, RegexTokenizer chi tiết hơn.

