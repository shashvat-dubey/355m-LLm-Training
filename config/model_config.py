from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Vocabulary
    vocab_size: int = 50_257

    # Maximum sequence length
    context_length: int = 1_024

    # Transformer architecture
    n_layers: int = 24
    n_heads: int = 16
    d_model: int = 1_024

    # Feed-forward network
    mlp_ratio: int = 4

    # Regularization
    dropout: float = 0.0

    # Linear layer bias
    bias: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            "d_model must be divisible by n_heads"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden_dim(self) -> int:
        return self.d_model * self.mlp_ratio