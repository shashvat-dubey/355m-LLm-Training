import math

import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Create a linear-warmup + cosine-decay learning-rate scheduler.

    Learning rate:
        0 -> peak_lr during warmup
        peak_lr -> peak_lr * min_lr_ratio during decay
    """

    assert warmup_steps < max_steps, (
        "warmup_steps must be smaller than max_steps"
    )

    def lr_lambda(step: int) -> float:

        # -------------------------
        # Warmup
        # -------------------------
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # -------------------------
        # Cosine decay
        # -------------------------
        progress = (
            step - warmup_steps
        ) / (
            max_steps - warmup_steps
        )

        progress = min(max(progress, 0.0), 1.0)

        cosine_decay = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio)
            * cosine_decay
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


if __name__ == "__main__":
    from config.model_config import GPTConfig
    from model.gpt import GPT
    from training.optimizer import create_optimizer

    config = GPTConfig()

    model = GPT(config)

    optimizer = create_optimizer(
        model,
        learning_rate=3e-4,
    )

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=10,
        max_steps=100,
    )

    print("Learning-rate schedule:\n")

    for step in range(101):

        if step in [0, 1, 5, 10, 25, 50, 75, 100]:
            print(
                f"Step {step:3d} | "
                f"LR {optimizer.param_groups[0]['lr']:.8f}"
            )

        optimizer.step()
        scheduler.step()