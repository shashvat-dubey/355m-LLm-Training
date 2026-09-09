import torch

from model.gpt import GPT


def _get_parameter_groups(model):
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            name.endswith("bias")
            or "ln_" in name
            or name.endswith("ln_f.weight")
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return [
        {
            "params": decay_parameters,
            "weight_decay": 0.1,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]


def create_optimizer(
    model: GPT,
    optimizer_name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    use_fused: bool = True,
):
    optimizer_name = optimizer_name.lower()

    parameter_groups = _get_parameter_groups(model)

    # Set requested weight decay.
    parameter_groups[0]["weight_decay"] = weight_decay

    if optimizer_name == "adamw":

        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": betas,
            "eps": eps,
        }

        if (
            use_fused
            and torch.cuda.is_available()
        ):
            try:
                optimizer = torch.optim.AdamW(
                    parameter_groups,
                    fused=True,
                    **optimizer_kwargs,
                )

                print("Optimizer: AdamW (fused)")
                return optimizer

            except (TypeError, RuntimeError):
                pass

        optimizer = torch.optim.AdamW(
            parameter_groups,
            **optimizer_kwargs,
        )

        print("Optimizer: AdamW")
        return optimizer

    if optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )

        print("Optimizer: Adam")
        return optimizer

    if optimizer_name == "sgd":

        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
        )

        print("Optimizer: SGD")
        return optimizer

    if optimizer_name == "rmsprop":

        optimizer = torch.optim.RMSprop(
            parameter_groups,
            lr=learning_rate,
            eps=eps,
        )

        print("Optimizer: RMSprop")
        return optimizer

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. "
        "Use 'adamw', 'adam', 'sgd', or 'rmsprop'."
    )


if __name__ == "__main__":
    from config.model_config import GPTConfig

    config = GPTConfig(
        context_length=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
    )

    model = GPT(config)

    optimizer = create_optimizer(
        model=model,
        optimizer_name="adamw",
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )