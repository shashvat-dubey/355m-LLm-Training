import torch
import torch.nn.functional as F


def language_modeling_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    batch_size, sequence_length, vocab_size = logits.shape

    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    targets = targets.reshape(
        batch_size * sequence_length,
    )

    return F.cross_entropy(
        logits,
        targets,
    )


if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)

    targets = torch.randint(
        0,
        10,
        (2, 4),
    )

    loss = language_modeling_loss(
        logits,
        targets,
    )

    print("Loss:", loss.item())