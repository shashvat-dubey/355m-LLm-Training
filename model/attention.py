import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.out_proj._is_residual_projection = True

        self.dropout_probability = config.dropout

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_output = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=(
                    self.dropout_probability
                    if self.training
                    else 0.0
                ),
                is_causal=True,
            )
        )

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return self.out_proj(
            attention_output
        )


if __name__ == "__main__":
    config = GPTConfig(
        context_length=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
    )

    attention = CausalSelfAttention(config)

    x = torch.randn(
        2,
        128,
        128,
    )

    output = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)