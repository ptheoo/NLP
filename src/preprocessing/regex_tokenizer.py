import re
from typing import List
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Regex: \w+ = từ, [^\w\s] = ký tự không phải chữ hoặc khoảng trắng
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens
