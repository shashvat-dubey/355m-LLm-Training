import time

import torch
from torch.utils.data import DataLoader

from model.gpt import GPT
from training.loss import language_modeling_loss
from training.checkpoint import CheckpointManager
from evaluation.loss import evaluate_loss, perplexity
from utils.device import get_device
from utils.logging import TensorBoardLogger


class Trainer:
    def __init__(
        self,
        model: GPT,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
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

        # --------------------------------------------------------
        # AMP
        # --------------------------------------------------------

        self.use_amp = (
            use_amp
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.model.to(self.device)

        # --------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------

        self.checkpoint_manager = (
            checkpoint_manager
        )

        self.checkpoint_interval = (
            checkpoint_interval
        )

        # --------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------

        self.eval_dataloader = (
            eval_dataloader
        )

        self.eval_interval = (
            eval_interval
        )

        self.eval_batches = eval_batches

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        self.logger = logger

    def train_step(
        self,
        x,
        targets,
    ):
        """
        Run one forward/backward pass.
        """

        x = x.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
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

        if self.use_amp:
            self.scaler.scale(
                loss_for_backward
            ).backward()
        else:
            loss_for_backward.backward()

        return loss.detach()

    def optimizer_step(self):
        """
        Apply accumulated gradients.
        """

        if self.use_amp:
            self.scaler.unscale_(
                self.optimizer
            )

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        if self.use_amp:
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
        """
        Save current training state.
        """

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
        """
        Restore training state.
        """

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
        """
        Run validation and log metrics.
        """

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
        """
        Train for max_steps optimizer steps.
        """

        self.model.train()

        data_iterator = iter(
            self.dataloader
        )

        total_loss = 0.0

        tokens_per_microbatch = (
            self.dataloader.batch_size
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

            # ----------------------------------------------------
            # Gradient accumulation
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Optimizer update
            # ----------------------------------------------------

            self.optimizer_step()

            average_loss = (
                step_loss
                / self.gradient_accumulation_steps
            )

            total_loss += average_loss

            # ----------------------------------------------------
            # Performance
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Logging
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Evaluation
            # ----------------------------------------------------

            if (
                self.eval_dataloader is not None
                and self.eval_interval > 0
                and step % self.eval_interval == 0
            ):
                self.evaluate(step)

            # ----------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------

            if (
                self.checkpoint_manager
                is not None
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