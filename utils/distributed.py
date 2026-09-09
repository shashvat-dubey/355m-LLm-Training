import os

import torch
import torch.distributed as dist


def is_distributed():
    return (
        dist.is_available()
        and dist.is_initialized()
    )


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl"
    )

    return rank, local_rank, world_size


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def is_main_process():
    return (
        not is_distributed()
        or dist.get_rank() == 0
    )


def barrier():
    if is_distributed():
        dist.barrier()