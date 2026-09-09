from dataclasses import dataclass


@dataclass
class DataConfig:

    # Hugging Face dataset
    dataset_name: str = "HuggingFaceFW/fineweb-edu"

    # FineWeb-Edu configuration
    dataset_subset: str = "sample-10BT"

    # Stream instead of downloading everything
    streaming: bool = True

    # Number of workers for data loading
    num_workers: int = 0