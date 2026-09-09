from torch.utils.data import DataLoader

from config.data_config import DataConfig
from config.model_config import GPTConfig
from data.dataset import FineWebDataset


def create_dataloader(
    data_config: DataConfig,
    model_config: GPTConfig,
    batch_size: int,
) -> DataLoader:

    dataset = FineWebDataset(
        data_config=data_config,
        model_config=model_config,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    from config.data_config import DataConfig
    from config.model_config import GPTConfig

    data_config = DataConfig()
    model_config = GPTConfig()

    dataloader = create_dataloader(
        data_config=data_config,
        model_config=model_config,
        batch_size=2,
    )

    x, y = next(iter(dataloader))

    print("Input batch shape:", x.shape)
    print("Target batch shape:", y.shape)