import os
import glob
from typing import Optional

import torch


class CheckpointManager:
    def __init__(
        self,
        directory: str = "checkpoints",
        max_checkpoints: int = 2,
    ):
        self.directory = directory
        self.max_checkpoints = max_checkpoints

        os.makedirs(
            self.directory,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        step: int = 0,
        loss: Optional[float] = None,
    ):
        """
        Save a complete training checkpoint.
        """

        checkpoint_path = os.path.join(
            self.directory,
            f"checkpoint_{step}.pt",
        )

        temporary_path = (
            checkpoint_path + ".tmp"
        )

        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint[
                "scheduler_state_dict"
            ] = scheduler.state_dict()

        if scaler is not None:
            checkpoint[
                "scaler_state_dict"
            ] = scaler.state_dict()

        if loss is not None:
            checkpoint["loss"] = loss

        # Save to a temporary file first.
        torch.save(
            checkpoint,
            temporary_path,
        )

        # Only replace the real checkpoint after
        # the save operation completed successfully.
        os.replace(
            temporary_path,
            checkpoint_path,
        )

        self._cleanup_old_checkpoints()

        print(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )

        return checkpoint_path

    def load(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device="cpu",
    ):
        """
        Load a checkpoint and restore training state.

        Returns:
            step, loss
        """

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        if (
            scaler is not None
            and "scaler_state_dict" in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        step = checkpoint.get(
            "step",
            0,
        )

        loss = checkpoint.get(
            "loss",
            None,
        )

        print(
            f"Checkpoint loaded: {path}"
        )

        print(
            f"Resuming from step: {step}"
        )

        if loss is not None:
            print(
                f"Checkpoint loss: {loss:.4f}"
            )

        return step, loss

    def get_latest_checkpoint(self):
        """
        Return the most recent checkpoint,
        or None if no checkpoints exist.
        """

        checkpoints = glob.glob(
            os.path.join(
                self.directory,
                "checkpoint_*.pt",
            )
        )

        if not checkpoints:
            return None

        checkpoints.sort(
            key=os.path.getmtime
        )

        return checkpoints[-1]

    def _cleanup_old_checkpoints(self):
        """
        Keep only the newest max_checkpoints files.
        """

        checkpoints = glob.glob(
            os.path.join(
                self.directory,
                "checkpoint_*.pt",
            )
        )

        checkpoints.sort(
            key=os.path.getmtime,
            reverse=True,
        )

        old_checkpoints = checkpoints[
            self.max_checkpoints:
        ]

        for path in old_checkpoints:
            os.remove(path)

            print(
                f"Removed old checkpoint: "
                f"{path}"
            )


if __name__ == "__main__":
    from config.model_config import GPTConfig
    from model.gpt import GPT
    from training.optimizer import create_optimizer
    from training.scheduler import create_scheduler

    config = GPTConfig()

    model = GPT(config)

    optimizer = create_optimizer(
        model=model,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=100,
    )

    manager = CheckpointManager(
        directory="checkpoints_test",
        max_checkpoints=2,
    )

    print("\nSaving checkpoints...\n")

    for step in [100, 200, 300]:

        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=5.0,
        )

    print("\nAvailable checkpoints:")

    checkpoints = glob.glob(
        "checkpoints_test/checkpoint_*.pt"
    )

    for checkpoint in checkpoints:
        print(checkpoint)

    latest = (
        manager.get_latest_checkpoint()
    )

    print(
        "\nLatest checkpoint:",
        latest,
    )

    print(
        "\nTesting checkpoint loading...\n"
    )

    manager.load(
        path=latest,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    print(
        "\nCheckpoint test complete."
    )