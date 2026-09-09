import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

from config.data_config import DataConfig
from config.model_config import GPTConfig
from data.tokenizer import GPTTokenizer


class FineWebDataset(IterableDataset):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: GPTConfig,
    ):
        super().__init__()

        self.data_config = data_config
        self.context_length = model_config.context_length

        self.tokenizer = GPTTokenizer()

    def _load_dataset(self):
        return load_dataset(
            self.data_config.dataset_name,
            name=self.data_config.dataset_subset,
            split="train",
            streaming=self.data_config.streaming,
        )

    def __iter__(self):

        dataset = self._load_dataset()

        # Store unused tokens between documents
        token_buffer = []

        for example in dataset:

            text = example["text"]

            # Convert text → token IDs
            tokens = self.tokenizer.encode(text)

            # Mark document boundary
            tokens.append(
                self.tokenizer.encoding.eot_token
            )

            # Add tokens to our continuous stream
            token_buffer.extend(tokens)

            # Create fixed-length training examples
            while len(token_buffer) >= self.context_length + 1:

                # Grab enough tokens for input + target
                chunk = token_buffer[
                    :self.context_length + 1
                ]

                # Remove the consumed tokens
                token_buffer = token_buffer[
                    self.context_length:
                ]

                x = torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                )

                y = torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                )

                yield x, y


if __name__ == "__main__":
    from config.data_config import DataConfig
    from config.model_config import GPTConfig

    data_config = DataConfig()
    model_config = GPTConfig()

    dataset = FineWebDataset(
        data_config=data_config,
        model_config=model_config,
    )

    iterator = iter(dataset)

    x, y = next(iterator)

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print("\nFirst 20 input tokens:")
    print(x[:20].tolist())

    print("\nFirst 20 target tokens:")
    print(y[:20].tolist())