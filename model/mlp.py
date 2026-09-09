import torch
import torch.nn as nn

from config.model_config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.fc_in = nn.Linear(
            config.d_model,
            config.mlp_hidden_dim,
            bias=config.bias,
        )

        self.activation = nn.GELU()

        self.fc_out = nn.Linear(
            config.mlp_hidden_dim,
            config.d_model,
            bias=config.bias,
        )

        self.fc_out._is_residual_projection = True

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_out(x)
        x = self.dropout(x)

        return x