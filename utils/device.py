import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)

    return "CPU"


if __name__ == "__main__":
    device = get_device()

    print("Device:", device)
    print("Device name:", get_device_name(device))

    if device.type == "cuda":
        print(
            "CUDA version:",
            torch.version.cuda,
        )

        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB",
        )