from lab1_2.src.preprocessing.simple_tokenizer import SimpleTokenizer
from lab1_2.src.preprocessing.regex_tokenizer import RegexTokenizer
from lab1_2.src.core.dataset_loaders import load_raw_text_data

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Đường dẫn dataset 
    dataset_path = "C:/Users/ok/Documents/NLP/UD_English-EWT/en_ewt-ud-dev.conllu"

    raw_text = load_raw_text_data(dataset_path)

    # Lấy mẫu 500 ký tự đầu tiên
    sample_text = raw_text[:500]
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
