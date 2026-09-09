import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config.model_config import GPTConfig
from config.data_config import DataConfig
from config.training_config import TrainingConfig

from model.gpt import GPT

from data.dataloader import create_dataloader

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer
from training.checkpoint import CheckpointManager

from utils.logging import TensorBoardLogger


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

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def main():

    rank, local_rank, world_size = (
        setup_distributed()
    )

    is_main_process = rank == 0

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    model_config = GPTConfig()

    data_config = DataConfig()

    training_config = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=8,

        precision="fp16",
        use_tf32=False,

        max_steps=100_000,

        learning_rate=3e-4,
        weight_decay=0.1,

        beta1=0.9,
        beta2=0.95,
        eps=1e-8,

        warmup_steps=2_000,
        min_lr_ratio=0.1,

        max_grad_norm=1.0,

        use_fused_optimizer=True,

        log_interval=10,

        eval_interval=0,
        checkpoint_interval=0,
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        f"cuda:{local_rank}"
    )

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    if is_main_process:

        print("=" * 60)
        print("355M GPT TRAINING")
        print("=" * 60)

        print(
            f"World size: {world_size}"
        )

        for index in range(
            torch.cuda.device_count()
        ):
            print(
                f"GPU {index}: "
                f"{torch.cuda.get_device_name(index)}"
            )

        print("=" * 60)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GPT(model_config)

    model.to(device)

    if is_main_process:

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"Parameters: "
            f"{parameter_count:,}"
        )

    # --------------------------------------------------
    # DDP
    # --------------------------------------------------

    if world_size > 1:

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    dataloader = create_dataloader(
        data_config=data_config,
        model_config=model_config,
        batch_size=training_config.batch_size,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        optimizer_name=training_config.optimizer,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=(
            training_config.beta1,
            training_config.beta2,
        ),
        eps=training_config.eps,
        use_fused=training_config.use_fused_optimizer,
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
        min_lr_ratio=training_config.min_lr_ratio,
    )

    # --------------------------------------------------
    # Checkpoints
    # --------------------------------------------------

    checkpoint_manager = None

    if is_main_process:

        checkpoint_manager = CheckpointManager(
            checkpoint_dir="checkpoints",
            max_checkpoints=2,
        )

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------

    logger = None

    if is_main_process:

        logger = TensorBoardLogger(
            log_dir="runs/355m_gpt"
        )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,

        gradient_accumulation_steps=(
            training_config.gradient_accumulation_steps
        ),

        max_grad_norm=(
            training_config.max_grad_norm
        ),

        precision=training_config.precision,
        use_tf32=training_config.use_tf32,

        checkpoint_manager=checkpoint_manager,

        checkpoint_interval=(
            training_config.checkpoint_interval
        ),

        eval_dataloader=None,

        eval_interval=(
            training_config.eval_interval
        ),

        logger=logger,

        log_interval=(
            training_config.log_interval
        ),

        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
    )

    # --------------------------------------------------
    # Synchronize
    # --------------------------------------------------

    if world_size > 1:
        dist.barrier()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    trainer.train(
        max_steps=training_config.max_steps
    )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    if logger is not None:
        logger.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()