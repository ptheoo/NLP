import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        # Chuyển về lowercase
        text = text.lower()
        # Tách dấu câu ra khỏi từ (.,?!)
        text = re.sub(r'([.,?!])', r' \1 ', text)
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        # Split theo khoảng trắng
        tokens = text.split(" ")
        return tokens
