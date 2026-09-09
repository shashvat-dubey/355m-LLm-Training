import torch
import torch.nn as nn

from config.model_config import GPTConfig
from model.embeddings import GPTEmbeddings
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # --------------------------------------------------
        # Embeddings
        # --------------------------------------------------

        self.embeddings = GPTEmbeddings(config)

        # --------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        self.ln_f = nn.LayerNorm(config.d_model)

        # --------------------------------------------------
        # Language-modeling head
        # --------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # --------------------------------------------------
        # GPT-style initialization
        # --------------------------------------------------

        self.apply(self._init_weights)

        # --------------------------------------------------
        # Weight tying
        # --------------------------------------------------

        self.lm_head.weight = (
            self.embeddings.token_embeddings.weight
        )

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:

        if isinstance(module, nn.Linear):

            # Standard GPT-style initialization
            std = 0.02

            # Smaller initialization for projections that
            # feed directly into residual connections.
            if getattr(
                module,
                "_is_residual_projection",
                False,
            ):
                std *= (
                    2 * self.config.n_layers
                ) ** -0.5

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------
        # Input embeddings
        # --------------------------------------------------

        x = self.embeddings(token_ids)

        # --------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        x = self.ln_f(x)

        # --------------------------------------------------
        # Vocabulary logits
        # --------------------------------------------------

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    config = GPTConfig()

    model = GPT(config)

    token_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 16),
    )

    logits = model(token_ids)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Input shape:", token_ids.shape)
    print("Output shape:", logits.shape)
    print(f"Parameters: {parameter_count:,}")