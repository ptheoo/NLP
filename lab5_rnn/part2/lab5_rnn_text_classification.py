"""
lab5_rnn_solution.py — phiên bản tự động trỏ đúng thư mục NLP/data/hwu
"""

import os
import tarfile
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================
# 1) Giải nén (không cần dùng nữa)
# ==========================
def extract_data(tar_path, dest_dir):
    if not os.path.exists(tar_path):
        print(f"Không tìm thấy file {tar_path}, bỏ qua giải nén.")
        return
    print(f"Đang giải nén {tar_path} ...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(dest_dir)
    print("Giải nén xong.")


# ==========================
# 2) Load data — chỉnh để trỏ NLP/data/hwu
# ==========================
def load_data():
    """
    Tự động lấy đúng đường dẫn NLP/data/hwu
    """
    this_file = os.path.abspath(__file__)
    part2_dir = os.path.dirname(this_file)
    lab5_rnn_dir = os.path.dirname(part2_dir)
    project_root = os.path.dirname(lab5_rnn_dir)
    data_dir = os.path.join(project_root, "data", "hwu")

    print(f"[DEBUG] Using data folder: {data_dir}")

    paths = {
        "train": os.path.join(data_dir, "train.csv"),
        "val":   os.path.join(data_dir, "val.csv"),
        "test":  os.path.join(data_dir, "test.csv"),
    }

    for split, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"❌ Không tìm thấy file: {p}\n"
                f"👉 Hãy kiểm tra thư mục: {data_dir}"
            )
        print(f"[OK] Found {split}: {p}")

    df_train = pd.read_csv(paths["train"])
    df_val = pd.read_csv(paths["val"])
    df_test = pd.read_csv(paths["test"])

    return df_train, df_val, df_test


# ==========================
# 3) Encode labels
# ==========================
def encode_labels(df_train, df_val, df_test):
    le = LabelEncoder()
    le.fit(pd.concat([df_train["category"], df_val["category"], df_test["category"]]))
    
    y_train = le.transform(df_train["category"])
    y_val   = le.transform(df_val["category"])
    y_test  = le.transform(df_test["category"])
    
    return le, y_train, y_val, y_test, len(le.classes_)


# ==========================
# 4) Task 1 — TF-IDF + Logistic Regression
# ==========================
def task1_tfidf_logreg(df_train, df_test, y_train, y_test):
    print("\n--- Task 1: TF-IDF + Logistic Regression ---")

    clf = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000)
    )

    clf.fit(df_train["text"], y_train)
    pred = clf.predict(df_test["text"])

    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    return clf


# ==========================
# 5) Train Word2Vec
# ==========================
def train_word2vec(sentences, vector_size=100):
    print("Đang train Word2Vec ...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        seed=SEED
    )

    model.train(sentences, total_examples=len(sentences), epochs=20)
    print("Word2Vec xong.")

    return model


def sentence_to_vec(text, w2v):
    tokens = str(text).split()
    vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]

    if len(vectors) == 0:
        return np.zeros(w2v.vector_size)

    return np.mean(vectors, axis=0)


# ==========================
# 6) Task 2 — Word2Vec average + Dense
# ==========================
def task2(df_train, df_val, df_test, y_train, y_val, y_test):
    print("\n--- Task 2: Word2Vec average + Dense ---")

    sentences = [s.split() for s in df_train["text"]]
    w2v = train_word2vec(sentences)

    X_train = np.vstack([sentence_to_vec(s, w2v) for s in df_train["text"]])
    X_val = np.vstack([sentence_to_vec(s, w2v) for s in df_val["text"]])
    X_test = np.vstack([sentence_to_vec(s, w2v) for s in df_test["text"]])

    num_classes = len(np.unique(y_train))

    model = Sequential([
        Dense(128, activation="relu", input_shape=(w2v.vector_size,)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=32, callbacks=[es], verbose=2)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy:", acc)

    return w2v, model


# ==========================
# 7) Task 3 — Pretrained embedding + LSTM
# ==========================
def build_embedding_matrix(tokenizer, w2v, dim):
    vocab_size = len(tokenizer.word_index) + 1
    M = np.zeros((vocab_size, dim))

    for word, idx in tokenizer.word_index.items():
        if word in w2v.wv:
            M[idx] = w2v.wv[word]

    return M


def task3(df_train, df_val, df_test, y_train, y_val, y_test, w2v):
    print("\n--- Task 3: Pretrained embedding + LSTM ---")

    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(pd.concat([df_train["text"], df_val["text"], df_test["text"]]))

    max_len = 50

    def encode(df):
        seq = tokenizer.texts_to_sequences(df["text"])
        return pad_sequences(seq, maxlen=max_len, padding="post")

    X_train = encode(df_train)
    X_val = encode(df_val)
    X_test = encode(df_test)

    emb_dim = w2v.vector_size
    emb_matrix = build_embedding_matrix(tokenizer, w2v, emb_dim)

    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, emb_dim,
                  weights=[emb_matrix], input_length=max_len, trainable=False),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(len(np.unique(y_train)), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=32, callbacks=[es], verbose=2)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy:", acc)

    return tokenizer, max_len, model


# ==========================
# 8) Task 4 — Scratch embedding + LSTM
# ==========================
def task4(tokenizer, max_len, df_train, df_val, df_test, y_train, y_val, y_test):
    print("\n--- Task 4: LSTM + embedding scratch ---")

    def encode(df):
        seq = tokenizer.texts_to_sequences(df["text"])
        return pad_sequences(seq, maxlen=max_len, padding="post")

    X_train = encode(df_train)
    X_val = encode(df_val)
    X_test = encode(df_test)

    vocab = len(tokenizer.word_index) + 1

    model = Sequential([
        Embedding(vocab, 100, input_length=max_len),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(len(np.unique(y_train)), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=32, callbacks=[es], verbose=2)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy:", acc)

    return model


# ==========================
# MAIN — chạy full pipeline
# ==========================
def main():
    df_train, df_val, df_test = load_data()
    le, y_train, y_val, y_test, _ = encode_labels(df_train, df_val, df_test)

    task1_tfidf_logreg(df_train, df_test, y_train, y_test)

    # Task 2
    try:
        w2v, _ = task2(df_train, df_val, df_test, y_train, y_val, y_test)
    except Exception as e:
        print("Task 2 lỗi:", e)
        w2v = None

    # Task 3
    if w2v is not None:
        try:
            tokenizer, max_len, _ = task3(df_train, df_val, df_test, y_train, y_val, y_test, w2v)
        except Exception as e:
            print("Task 3 lỗi:", e)
            tokenizer = None
            max_len = 50
    else:
        tokenizer = None
        max_len = 50

    # Task 4
    if tokenizer is not None:
        task4(tokenizer, max_len, df_train, df_val, df_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
