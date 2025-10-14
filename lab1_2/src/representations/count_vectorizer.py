from lab1_2.src.core.interfaces import Vectorizer
from lab1_2.src.core.interfaces import Tokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)

        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        return self

    def transform(self, documents: list[str]) -> list[list[int]]:
        vectors = []
        for doc in documents:
            vector = [0] * len(self.vocabulary_)
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    vector[idx] += 1
            vectors.append(vector)
        return vectors

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)
