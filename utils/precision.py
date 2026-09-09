from contextlib import nullcontext

import torch


def configure_tf32(enabled: bool = True):
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def get_amp_dtype(precision: str):
    precision = precision.lower()

    if precision == "fp16":
        return torch.float16

    if precision == "bf16":
        return torch.bfloat16

    if precision == "fp32":
        return torch.float32

    raise ValueError(
        f"Unsupported precision: {precision}. "
        "Use 'fp32', 'fp16', or 'bf16'."
    )


def autocast_context(
    device: torch.device,
    precision: str,
):
    precision = precision.lower()

    if device.type != "cuda":
        return nullcontext()

    if precision == "fp16":
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 was requested, but this GPU does not "
                "appear to support BF16."
            )

        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    if precision == "fp32":
        return nullcontext()

    raise ValueError(
        f"Unsupported precision: {precision}"
    )


def use_grad_scaler(
    device: torch.device,
    precision: str,
) -> bool:
    return (
        device.type == "cuda"
        and precision.lower() == "fp16"
    )