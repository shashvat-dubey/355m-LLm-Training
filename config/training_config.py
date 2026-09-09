from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Batch
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Optimizer
    optimizer: str = "adamw"

    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Training
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    min_lr_ratio: float = 0.1

    max_grad_norm: float = 1.0

    # Precision
    precision: str = "bf16"

    # NVIDIA TF32
    use_tf32: bool = True

    # Fused optimizer implementation when available
    use_fused_optimizer: bool = True

    # Optional activation checkpointing
    use_gradient_checkpointing: bool = False

    # Logging
    log_interval: int = 10
    eval_interval: int = 0
    checkpoint_interval: int = 0