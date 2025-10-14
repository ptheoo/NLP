from lab1_2.src.preprocessing.simple_tokenizer import SimpleTokenizer
from lab1_2.src.preprocessing.regex_tokenizer import RegexTokenizer

if __name__ == "__main__":
    simple_tok = SimpleTokenizer()
    regex_tok = RegexTokenizer()

    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    for sent in sentences:
        print("\nSentence:", sent)
        print("SimpleTokenizer:", simple_tok.tokenize(sent))
        print("RegexTokenizer:", regex_tok.tokenize(sent))
