import torch

from model.gpt import GPT
from training.loss import language_modeling_loss


@torch.no_grad()
def evaluate_loss(
    model: GPT,
    dataloader,
    device: torch.device,
    max_batches: int = 100,
):
    """
    Evaluate average language-modeling loss
    over a fixed number of batches.
    """

    model.eval()

    total_loss = 0.0
    batches = 0

    for x, targets in dataloader:

        x = x.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(x)

        loss = language_modeling_loss(
            logits,
            targets,
        )

        total_loss += loss.item()
        batches += 1

        if batches >= max_batches:
            break

    model.train()

    if batches == 0:
        raise RuntimeError(
            "Evaluation dataloader produced no batches."
        )

    return total_loss / batches


def perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.

    Perplexity = exp(loss)
    """

    return torch.exp(
        torch.tensor(loss)
    ).item()


if __name__ == "__main__":
    from config.model_config import GPTConfig
    from config.data_config import DataConfig
    from data.dataloader import create_dataloader
    from utils.device import get_device

    config = GPTConfig(
        context_length=128,
        n_layers=4,
        n_heads=4,
        d_model=128,
    )

    data_config = DataConfig()

    device = get_device()

    model = GPT(config)
    model.to(device)

    dataloader = create_dataloader(
        data_config=data_config,
        model_config=config,
        batch_size=1,
    )

    loss = evaluate_loss(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=2,
    )

    ppl = perplexity(loss)

    print("Evaluation loss:", loss)
    print("Perplexity:", ppl)