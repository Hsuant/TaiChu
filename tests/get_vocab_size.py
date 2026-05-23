from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./taichu_tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(vocab_size)