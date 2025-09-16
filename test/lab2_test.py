from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.core.dataset_loaders import load_raw_text_data


def test_count_vectorizer():
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    X = vectorizer.fit_transform(corpus)

    print("=== Corpus ===")
    for doc in corpus:
        print(doc)

    print("\n=== Learned Vocabulary ===")
    for token, idx in vectorizer.vocabulary_.items():
        print(f"{token}: {idx}")

    print("\n=== Document-Term Matrix ===")
    for row in X:
        print(row)


def test_with_ud_dataset():
    dataset_path = r"C:\Users\ok\Documents\NLP\UD_English-EWT\en_ewt-ud-dev.conllu"
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    raw_text = load_raw_text_data(dataset_path)
    sample_texts = [
        raw_text[:200],
        raw_text[200:400],
        raw_text[400:600]
    ]

    X = vectorizer.fit_transform(sample_texts)

    print("\n=== Sample Corpus from UD-English-EWT ===")
    for doc in sample_texts:
        print(doc, "\n")

    print("\n=== Learned Vocabulary (UD Sample) ===")
    for token, idx in vectorizer.vocabulary_.items():
        print(f"{token}: {idx}")

    print("\n=== Document-Term Matrix (UD Sample) ===")
    for row in X:
        print(row)


if __name__ == "__main__":
    test_count_vectorizer()
    test_with_ud_dataset()
