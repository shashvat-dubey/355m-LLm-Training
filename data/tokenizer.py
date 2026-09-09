import tiktoken


class GPTTokenizer:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    @property
    def eot_token_id(self) -> int:
        return self.encoding.eot_token

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)


if __name__ == "__main__":
    tokenizer = GPTTokenizer()

    text = "Hello! We are building a language model from scratch."

    token_ids = tokenizer.encode(text)

    print("Original text:")
    print(text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nDecoded text:")
    print(tokenizer.decode(token_ids))

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)