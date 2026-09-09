import torch
import torch.nn as nn

from config.model_config import GPTConfig
from model.attention import CausalSelfAttention
from model.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.ln_2 = nn.LayerNorm(
            config.d_model
        )

        self.mlp = MLP(
            config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # First residual path
        x = x + self.attention(
            self.ln_1(x)
        )

        # Second residual path
        x = x + self.mlp(
            self.ln_2(x)
        )

        return x