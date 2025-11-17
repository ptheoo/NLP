"""
analyze_qualitative.py
Mục đích: phân tích định tính mô hình text classification trên các câu "khó"
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from part2.lab5_rnn_text_classification import main

# Khởi tạo tất cả mô hình
clf1, model2, w2v, tokenizer, max_len, model3, model4, le = main()

# Ví dụ các câu "khó"
examples = [
    ("can you remind me to not call my mom", "reminder_create"),
    ("is it going to be sunny or rainy tomorrow", "weather_query"),
    ("find a flight from new york to london but not through paris", "flight_search")
]

# ----------------------
# Hàm encode cho Word2Vec + Dense
# ----------------------
def sentence_to_vec(text, w2v):
    tokens = text.split()
    vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if len(vectors) == 0:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)

# ----------------------
# Hàm encode cho LSTM
# ----------------------
def encode_text(texts, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post")

# ----------------------
# Chạy dự đoán và in bảng
def qualitative_analysis():
    print("=== Phân tích định tính ===\n")
    for text, true_label in examples:
        print(f"Text: {text}")

        # Task 1 — TF-IDF + LR
        pred1 = clf1.predict([text])
        pred1_label = le.inverse_transform(pred1)[0]

        # Task 2 — Word2Vec + Dense
        vec = sentence_to_vec(text, w2v)
        pred2 = model2.predict(np.expand_dims(vec, axis=0))
        pred2_label = le.inverse_transform([np.argmax(pred2)])[0]

        # Task 3 — Pre-trained LSTM
        X3 = encode_text([text], tokenizer, max_len)
        pred3 = model3.predict(X3)
        pred3_label = le.inverse_transform([np.argmax(pred3)])[0]

        # Task 4 — Scratch LSTM
        pred4 = model4.predict(X3)
        pred4_label = le.inverse_transform([np.argmax(pred4)])[0]

        # In bảng kết quả so với nhãn thật
        print(f"{'Mô hình':<20} | {'Dự đoán':<20} | {'Nhãn thật':<20}")
        print(f"{'-'*60}")
        print(f"{'TF-IDF + LR':<20} | {pred1_label:<20} | {true_label:<20}")
        print(f"{'Word2Vec + Dense':<20} | {pred2_label:<20} | {true_label:<20}")
        print(f"{'Pre-trained LSTM':<20} | {pred3_label:<20} | {true_label:<20}")
        print(f"{'Scratch LSTM':<20} | {pred4_label:<20} | {true_label:<20}")
        print("-"*60)


if __name__ == "__main__":
    qualitative_analysis()
