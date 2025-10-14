from abc import ABC, abstractmethod
from typing import List

# ========== Lab 1 ==========
class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Chuyển văn bản thành danh sách token"""
        pass


# ========== Lab 2 ==========
class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]):
        """
        Học từ vựng từ một danh sách document (corpus).
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Chuyển đổi danh sách document thành danh sách vector đếm
        dựa trên vocabulary đã học.
        """
        pass

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Tiện lợi: fit và transform cùng lúc.
        """
        self.fit(corpus)
        return self.transform(corpus)
