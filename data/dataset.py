import torch
from torch.utils.data import IterableDataset, get_worker_info
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

        # --------------------------------------------------
        # Distributed rank
        # --------------------------------------------------

        rank = 0
        world_size = 1

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # --------------------------------------------------
        # DataLoader worker information
        # --------------------------------------------------

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # --------------------------------------------------
        # Treat every GPU + worker as one data worker
        # --------------------------------------------------

        global_worker_id = (
            rank * num_workers + worker_id
        )

        total_workers = (
            world_size * num_workers
        )

        # --------------------------------------------------
        # Prefer HuggingFace streaming sharding
        # --------------------------------------------------

        try:
            dataset = dataset.shard(
                num_shards=total_workers,
                index=global_worker_id,
            )

            use_shard = True

        except AttributeError:
            use_shard = False

        # --------------------------------------------------
        # Token buffer
        # --------------------------------------------------

        token_buffer = []

        for example_index, example in enumerate(dataset):

            # --------------------------------------------------
            # Fallback sharding
            # --------------------------------------------------

            if not use_shard:

                if (
                    example_index % total_workers
                    != global_worker_id
                ):
                    continue

            # --------------------------------------------------
            # Tokenize
            # --------------------------------------------------

            text = example["text"]

            tokens = self.tokenizer.encode(text)

            tokens.append(
                self.tokenizer.eot_token_id
            )

            token_buffer.extend(tokens)

            # --------------------------------------------------
            # Produce fixed-length sequences
            # --------------------------------------------------

            while (
                len(token_buffer)
                >= self.context_length + 1
            ):

                chunk = token_buffer[
                    : self.context_length + 1
                ]

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