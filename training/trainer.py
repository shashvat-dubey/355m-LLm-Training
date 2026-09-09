import time

import torch
from torch.utils.data import DataLoader

from model.gpt import GPT
from training.loss import language_modeling_loss
from evaluation.loss import evaluate_loss, perplexity
from utils.device import get_device
from utils.logging import TensorBoardLogger
from utils.precision import (
    autocast_context,
    configure_tf32,
    use_grad_scaler,
)


class Trainer:
    def __init__(
        self,
        model: GPT,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        precision: str = "fp32",
        use_tf32: bool = True,
        checkpoint_manager=None,
        checkpoint_interval: int = 0,
        eval_dataloader=None,
        eval_interval: int = 0,
        eval_batches: int = 10,
        logger=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        self.max_grad_norm = max_grad_norm

        self.device = get_device()

        self.precision = precision.lower()

        if self.precision not in {
            "fp32",
            "fp16",
            "bf16",
        }:
            raise ValueError(
                f"Unsupported precision: {self.precision}"
            )

        configure_tf32(use_tf32)

        self.use_scaler = use_grad_scaler(
            self.device,
            self.precision,
        )

        self.scaler = None

        if self.use_scaler:
            self.scaler = torch.amp.GradScaler(
                "cuda"
            )

        self.model.to(self.device)

        self.checkpoint_manager = (
            checkpoint_manager
        )

        self.checkpoint_interval = (
            checkpoint_interval
        )

        self.eval_dataloader = eval_dataloader
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches

        self.logger = logger

        print(
            f"Precision: {self.precision}"
        )

        print(
            f"GradScaler: {self.use_scaler}"
        )

    def train_step(
        self,
        x,
        targets,
    ):
        x = x.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            non_blocking=True,
        )

        with autocast_context(
            self.device,
            self.precision,
        ):
            logits = self.model(x)

            loss = language_modeling_loss(
                logits,
                targets,
            )

        loss_for_backward = (
            loss
            / self.gradient_accumulation_steps
        )

        if self.use_scaler:
            self.scaler.scale(
                loss_for_backward
            ).backward()
        else:
            loss_for_backward.backward()

        return loss.detach()

    def optimizer_step(self):

        if self.use_scaler:
            self.scaler.unscale_(
                self.optimizer
            )

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        if self.use_scaler:
            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(
            set_to_none=True
        )

    def save_checkpoint(
        self,
        step: int,
        loss: float,
    ):
        if self.checkpoint_manager is None:
            return

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=step,
            loss=loss,
        )

    def resume_from_checkpoint(
        self,
        path: str,
    ):
        if self.checkpoint_manager is None:
            raise RuntimeError(
                "CheckpointManager is required."
            )

        step, loss = (
            self.checkpoint_manager.load(
                path=path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
            )
        )

        return step, loss

    def evaluate(
        self,
        step: int,
    ):
        if self.eval_dataloader is None:
            return None

        validation_loss = evaluate_loss(
            model=self.model,
            dataloader=self.eval_dataloader,
            device=self.device,
            max_batches=self.eval_batches,
        )

        validation_perplexity = perplexity(
            validation_loss
        )

        print(
            f"Step {step:6d} | "
            f"Val Loss {validation_loss:.4f} | "
            f"PPL {validation_perplexity:.2f}"
        )

        if self.logger is not None:
            self.logger.log_evaluation(
                step=step,
                loss=validation_loss,
                perplexity=validation_perplexity,
            )

            self.logger.flush()

        self.model.train()

        return validation_loss

    def train(
        self,
        max_steps: int,
        start_step: int = 0,
    ):
        self.model.train()

        data_iterator = iter(
            self.dataloader
        )

        total_loss = 0.0

        if self.dataloader.batch_size is None:
            batch_size = 1
        else:
            batch_size = (
                self.dataloader.batch_size
            )

        tokens_per_microbatch = (
            batch_size
            * self.model.config.context_length
        )

        tokens_per_step = (
            tokens_per_microbatch
            * self.gradient_accumulation_steps
        )

        for step in range(
            start_step + 1,
            max_steps + 1,
        ):
            step_start_time = time.time()

            step_loss = 0.0

            for _ in range(
                self.gradient_accumulation_steps
            ):
                try:
                    x, targets = next(
                        data_iterator
                    )

                except StopIteration:
                    data_iterator = iter(
                        self.dataloader
                    )

                    x, targets = next(
                        data_iterator
                    )

                loss = self.train_step(
                    x,
                    targets,
                )

                step_loss += loss.item()

            self.optimizer_step()

            average_loss = (
                step_loss
                / self.gradient_accumulation_steps
            )

            total_loss += average_loss

            step_time = (
                time.time()
                - step_start_time
            )

            tokens_seen = (
                step * tokens_per_step
            )

            learning_rate = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            if (
                step == start_step + 1
                or step % 10 == 0
            ):
                print(
                    f"Step {step:6d} | "
                    f"Loss {average_loss:.4f} | "
                    f"LR {learning_rate:.8f} | "
                    f"Time {step_time:.2f}s"
                )

            if self.logger is not None:
                self.logger.log_train(
                    step=step,
                    loss=average_loss,
                    learning_rate=learning_rate,
                    tokens_seen=tokens_seen,
                )

                self.logger.log_step_time(
                    step=step,
                    seconds=step_time,
                )

            if (
                self.eval_dataloader is not None
                and self.eval_interval > 0
                and step % self.eval_interval == 0
            ):
                self.evaluate(step)

            if (
                self.checkpoint_manager is not None
                and self.checkpoint_interval > 0
                and step % self.checkpoint_interval == 0
            ):
                self.save_checkpoint(
                    step=step,
                    loss=average_loss,
                )

            if self.logger is not None:
                self.logger.flush()

        steps_completed = (
            max_steps - start_step
        )

        if steps_completed <= 0:
            return 0.0

        return (
            total_loss
            / steps_completed
        )