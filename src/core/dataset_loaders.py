import os

def load_raw_text_data(path: str) -> str:
    """
    Load raw text data từ file UD (conllu hoặc txt).
    Kết hợp các từ lại thành một string để tokenizer xử lý.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    text_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split("\t")
            if len(parts) > 1:  
                text_data.append(parts[1])  # cột FORM (word)
    return " ".join(text_data)
