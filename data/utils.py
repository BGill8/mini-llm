import tiktoken

class TikTokenizer:
    def __init__(self, model_name="gpt2"):
        self.enc = tiktoken.get_encoding(model_name)
        self.vocab_size = self.enc.n_vocab
        self.pad_id = self.enc.eot_token

    def encode(self, text):
        return self.enc.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, tokens):
        return self.enc.decode(tokens)