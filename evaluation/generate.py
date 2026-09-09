import torch
from typing import Optional

from model.gpt import GPT
from data.tokenizer import GPTTokenizer

@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: GPTTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
):
    """
    Generate text autoregressively from a prompt.
    """

    model.eval()

    device = next(
        model.parameters()
    ).device

    # ------------------------------------------------------------
    # Encode prompt
    # ------------------------------------------------------------

    token_ids = tokenizer.encode(
        prompt
    )

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    # ------------------------------------------------------------
    # Generate one token at a time
    # ------------------------------------------------------------

    for _ in range(max_new_tokens):

        # Keep only the model's context window.
        input_ids_cond = input_ids[
            :, -model.config.context_length:
        ]

        # Forward pass.
        logits = model(
            input_ids_cond
        )

        # Only care about the final position.
        next_token_logits = logits[
            :, -1, :
        ]

        # --------------------------------------------------------
        # Temperature
        # --------------------------------------------------------

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0"
            )

        next_token_logits = (
            next_token_logits
            / temperature
        )

        # --------------------------------------------------------
        # Top-k filtering
        # --------------------------------------------------------

        if top_k is not None:

            top_k = min(
                top_k,
                next_token_logits.size(-1),
            )

            values, _ = torch.topk(
                next_token_logits,
                top_k,
            )

            minimum_value = values[
                :, -1
            ].unsqueeze(-1)

            next_token_logits = torch.where(
                next_token_logits
                < minimum_value,
                torch.tensor(
                    float("-inf"),
                    device=device,
                ),
                next_token_logits,
            )

        # --------------------------------------------------------
        # Convert logits → probabilities
        # --------------------------------------------------------

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        # --------------------------------------------------------
        # Sample next token
        # --------------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # Append token.
        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    # ------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------

    generated_tokens = (
        input_ids[0].tolist()
    )

    return tokenizer.decode(
        generated_tokens
    )


if __name__ == "__main__":
    from config.model_config import GPTConfig

    config = GPTConfig(
        vocab_size=50257,
        context_length=128,
        n_layers=4,
        n_heads=4,
        d_model=128,
        mlp_ratio=4,
        dropout=0.0,
        bias=True,
    )

    model = GPT(config)

    tokenizer = GPTTokenizer()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model.to(device)

    prompt = (
        "The future of artificial intelligence"
    )

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=50,
    )

    print("\nPrompt:")
    print(prompt)

    print("\nGenerated:")
    print(generated_text)