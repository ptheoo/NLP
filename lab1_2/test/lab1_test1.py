from lab1_2.src.preprocessing.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    simple_tok = SimpleTokenizer()
    sentence = "Hello, world! This is a test."
    print(simple_tok.tokenize(sentence))
