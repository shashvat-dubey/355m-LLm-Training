import time
from contextlib import nullcontext

import torch

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
        model,
        dataloader,
        optimizer,
        scheduler=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        precision="fp32",
        use_tf32=True,
        checkpoint_manager=None,
        checkpoint_interval=0,
        eval_dataloader=None,
        eval_interval=0,
        eval_batches=10,
        logger=None,
        local_rank=None,
        rank=0,
        world_size=1,
        log_interval=10,

    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.precision = precision.lower()

        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval = checkpoint_interval

        self.eval_dataloader = eval_dataloader
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches

        self.logger = logger
        self.log_interval = log_interval
        # Distributed information
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size

        self.is_main_process = self.rank == 0
        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = get_device(local_rank)

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        # --------------------------------------------------
        # Precision
        # --------------------------------------------------

        if self.precision not in {"fp32", "fp16", "bf16"}:
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
            self.scaler = torch.amp.GradScaler("cuda")

        # --------------------------------------------------
        # Move model
        # --------------------------------------------------

        self.model.to(self.device)

        # --------------------------------------------------
        # Initial state
        # --------------------------------------------------

        self.global_step = 0

        self.optimizer.zero_grad(set_to_none=True)

        if self.is_main_process:
            parameter_count = sum(
                parameter.numel()
                for parameter in self.model.parameters()
            )

            print(f"Device: {self.device}")
            print(f"World size: {self.world_size}")
            print(f"Parameters: {parameter_count:,}")
            print(
                f"Precision: {self.precision}"
            )
            print(
                f"GradScaler: {self.use_scaler}"
            )

    # ======================================================
    # Single microbatch
    # ======================================================

    def train_step(
        self,
        batch,
        sync_gradients=True,
    ):
        x, targets = batch

        x = x.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            non_blocking=True,
        )

        # --------------------------------------------------
        # DDP optimization
        #
        # During gradient accumulation we don't need to
        # synchronize gradients after every microbatch.
        # Only synchronize on the final microbatch.
        # --------------------------------------------------

        if (
            hasattr(self.model, "no_sync")
            and not sync_gradients
        ):
            ddp_context = self.model.no_sync()
        else:
            ddp_context = nullcontext()

        with ddp_context:

            with autocast_context(
                self.device,
                self.precision,
            ):
                logits = self.model(x)

                loss = language_modeling_loss(
                    logits,
                    targets,
                )

                loss = (
                    loss
                    / self.gradient_accumulation_steps
                )

            # --------------------------------------------------
            # Backward
            # --------------------------------------------------

            if self.scaler is not None:

                self.scaler.scale(loss).backward()

            else:

                loss.backward()

        return loss.detach()

    # ======================================================
    # Optimizer step
    # ======================================================

    def optimizer_step(self):

        # --------------------------------------------------
        # Unscale before gradient clipping
        # --------------------------------------------------

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
        )

        # --------------------------------------------------
        # Optimizer update
        # --------------------------------------------------

        if self.scaler is not None:

            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:

            self.optimizer.step()

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        if self.scheduler is not None:
            self.scheduler.step()

        # --------------------------------------------------
        # Clear gradients
        # --------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

    # ======================================================
    # Checkpoint
    # ======================================================

    def save_checkpoint(self, loss):

        # Only rank 0 writes checkpoints.
        if not self.is_main_process:
            return

        if self.checkpoint_manager is None:
            return

        model_to_save = self.model

        # DDP wrapper contains the real model in .module
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        self.checkpoint_manager.save(
            step=self.global_step,
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            loss=loss,
        )

    # ======================================================
    # Resume
    # ======================================================

    def resume_from_checkpoint(self, path):

        if self.checkpoint_manager is None:
            raise RuntimeError(
                "No checkpoint manager configured."
            )

        model_to_load = self.model

        if hasattr(model_to_load, "module"):
            model_to_load = model_to_load.module

        checkpoint = self.checkpoint_manager.load(
            path=path,
            model=model_to_load,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

        self.global_step = checkpoint["step"]

        if self.is_main_process:
            print(
                f"Resumed from step "
                f"{self.global_step}"
            )

        return checkpoint

    # ======================================================
    # Evaluation
    # ======================================================

    @torch.no_grad()
    def evaluate(self):

        if self.eval_dataloader is None:
            return None

        model_was_training = self.model.training

        self.model.eval()

        loss = evaluate_loss(
            self.model,
            self.eval_dataloader,
            self.device,
            max_batches=self.eval_batches,
        )

        # Restore training mode
        if model_was_training:
            self.model.train()

        # --------------------------------------------------
        # Average evaluation loss across GPUs
        # --------------------------------------------------

        loss_tensor = torch.tensor(
            loss,
            dtype=torch.float32,
            device=self.device,
        )

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(
                loss_tensor,
                op=torch.distributed.ReduceOp.SUM,
            )

            loss_tensor /= self.world_size

        loss = loss_tensor.item()

        if self.is_main_process:

            print(
                f"Step {self.global_step} | "
                f"Eval Loss {loss:.4f} | "
                f"Perplexity {perplexity(loss):.2f}"
            )

            if self.logger is not None:
                self.logger.log(
                    "eval/loss",
                    loss,
                    self.global_step,
                )

                self.logger.log(
                    "eval/perplexity",
                    perplexity(loss),
                    self.global_step,
                )

        return loss

    # ======================================================
    # Training
    # ======================================================

    def train(
        self,
        max_steps,
    ):

        data_iterator = iter(self.dataloader)

        # --------------------------------------------------
        # Determine batch size
        # --------------------------------------------------

        batch_size = self.dataloader.batch_size

        if batch_size is None:
            batch_size = 1

        model_config = (
            self.model.module.config
            if hasattr(self.model, "module")
            else self.model.config
        )

        context_length = model_config.context_length

        # --------------------------------------------------
        # Global token count
        # --------------------------------------------------

        tokens_per_microbatch = (
            batch_size
            * context_length
        )

        tokens_per_step = (
            tokens_per_microbatch
            * self.gradient_accumulation_steps
            * self.world_size
        )

        if self.is_main_process:

            print(
                f"Tokens per optimizer step: "
                f"{tokens_per_step:,}"
            )

            print(
                f"Starting training from step "
                f"{self.global_step + 1}"
            )

        # --------------------------------------------------
        # Main training loop
        # --------------------------------------------------

        while self.global_step < max_steps:

            start_time = time.time()

            total_loss = 0.0

            # ----------------------------------------------
            # Gradient accumulation
            # ----------------------------------------------

            for micro_step in range(
                self.gradient_accumulation_steps
            ):

                try:
                    batch = next(data_iterator)

                except StopIteration:
                    data_iterator = iter(
                        self.dataloader
                    )
                    batch = next(data_iterator)

                is_last_microbatch = (
                    micro_step
                    == self.gradient_accumulation_steps - 1
                )

                loss = self.train_step(
                    batch,
                    sync_gradients=is_last_microbatch,
                )

                total_loss += loss.item()

            # ----------------------------------------------
            # Optimizer update
            # ----------------------------------------------

            self.optimizer_step()

            self.global_step += 1

            elapsed = time.time() - start_time

            average_loss = total_loss

            # ----------------------------------------------
            # Learning rate
            # ----------------------------------------------

            if self.scheduler is not None:

                learning_rate = (
                    self.scheduler.get_last_lr()[0]
                )

            else:

                learning_rate = (
                    self.optimizer.param_groups[0]["lr"]
                )

            # ----------------------------------------------
            # Logging
            # ----------------------------------------------

            if (
                self.is_main_process
                and self.global_step % self.log_interval == 0
            ):

                tokens_per_second = (
                    tokens_per_step / elapsed
                )

                print(
                    f"Step {self.global_step:6d} | "
                    f"Loss {average_loss:.4f} | "
                    f"LR {learning_rate:.6g} | "
                    f"{tokens_per_second:,.0f} tok/s | "
                    f"{elapsed:.2f}s"
                )

                if self.logger is not None:

                    self.logger.log(
                        "train/loss",
                        average_loss,
                        self.global_step,
                    )

                    self.logger.log(
                        "train/learning_rate",
                        learning_rate,
                        self.global_step,
                    )

                    self.logger.log(
                        "train/tokens_per_second",
                        tokens_per_second,
                        self.global_step,
                    )

            # ----------------------------------------------
            # Evaluation
            # ----------------------------------------------

            if (
                self.eval_interval > 0
                and self.global_step % self.eval_interval == 0
            ):
                self.evaluate()

            # ----------------------------------------------
            # Checkpoint
            # ----------------------------------------------

            if (
                self.checkpoint_interval > 0
                and self.global_step
                % self.checkpoint_interval
                == 0
            ):
                self.save_checkpoint(
                    average_loss
                )

        if self.is_main_process:
            print(
                f"Training complete at step "
                f"{self.global_step}"
            )