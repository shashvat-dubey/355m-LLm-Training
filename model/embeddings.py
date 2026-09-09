import torch
import torch.nn as nn

from config.model_config import GPTConfig


class GPTEmbeddings(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape

        # Ensure the sequence fits within the model's context window
        assert sequence_length <= self.position_embeddings.num_embeddings, (
            f"Sequence length {sequence_length} exceeds "
            f"context length {self.position_embeddings.num_embeddings}"
        )

        # Create position IDs: [0, 1, 2, ..., sequence_length - 1]
        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embeddings(token_ids)
        position_vectors = self.position_embeddings(positions)

        x = token_vectors + position_vectors

        return self.dropout(x)