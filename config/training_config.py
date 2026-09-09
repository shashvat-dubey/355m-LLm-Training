from dataclasses import dataclass


@dataclass
class TrainingConfig:

    # --------------------------------------------------
    # Batch
    # --------------------------------------------------

    batch_size: int = 4

    gradient_accumulation_steps: int = 8

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer: str = "adamw"

    learning_rate: float = 3e-4

    weight_decay: float = 0.1

    beta1: float = 0.9
    beta2: float = 0.95

    eps: float = 1e-8

    # --------------------------------------------------
    # Training duration
    # --------------------------------------------------

    max_steps: int = 100_000

    warmup_steps: int = 2_000

    min_lr_ratio: float = 0.1

    # --------------------------------------------------
    # Stability
    # --------------------------------------------------

    max_grad_norm: float = 1.0

    # --------------------------------------------------
    # Precision
    # --------------------------------------------------

    precision: str = "fp16"

    # T4 does not have Ampere TF32 support.
    use_tf32: bool = False

    # Fused AdamW is available on CUDA.
    use_fused_optimizer: bool = True

    # 355M fits without checkpointing on T4.
    use_gradient_checkpointing: bool = False

    # --------------------------------------------------
    # Logging / evaluation
    # --------------------------------------------------

    log_interval: int = 10

    eval_interval: int = 0

    checkpoint_interval: int = 0