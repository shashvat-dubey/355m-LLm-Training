import os
import time

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(
        self,
        log_dir: str = "runs/gpt",
    ):
        self.log_dir = log_dir

        os.makedirs(
            self.log_dir,
            exist_ok=True,
        )

        self.writer = SummaryWriter(
            log_dir=self.log_dir
        )

        self.start_time = time.time()

    def log_train(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        tokens_seen: int,
    ):
        """
        Log training metrics.
        """

        self.writer.add_scalar(
            "train/loss",
            loss,
            step,
        )

        self.writer.add_scalar(
            "train/learning_rate",
            learning_rate,
            step,
        )

        self.writer.add_scalar(
            "train/tokens_seen",
            tokens_seen,
            step,
        )

    def log_evaluation(
        self,
        step: int,
        loss: float,
        perplexity: float,
    ):
        """
        Log evaluation metrics.
        """

        self.writer.add_scalar(
            "eval/loss",
            loss,
            step,
        )

        self.writer.add_scalar(
            "eval/perplexity",
            perplexity,
            step,
        )

    def log_step_time(
        self,
        step: int,
        seconds: float,
    ):
        """
        Log time required for one optimizer step.
        """

        self.writer.add_scalar(
            "performance/step_time",
            seconds,
            step,
        )

    def flush(self):
        """
        Force TensorBoard data to disk.
        """

        self.writer.flush()

    def close(self):
        """
        Close TensorBoard writer.
        """

        self.writer.close()


if __name__ == "__main__":
    logger = TensorBoardLogger(
        log_dir="runs/test"
    )

    print(
        "Writing test TensorBoard data..."
    )

    for step in range(1, 101):

        loss = 10.0 / (
            1.0 + step * 0.05
        )

        learning_rate = (
            3e-4
            * min(step / 20, 1.0)
        )

        tokens_seen = (
            step * 8192
        )

        logger.log_train(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            tokens_seen=tokens_seen,
        )

        logger.log_evaluation(
            step=step,
            loss=loss + 0.2,
            perplexity=2.71828 ** (
                loss + 0.2
            ),
        )

    logger.flush()
    logger.close()

    print(
        "\nTensorBoard test complete."
    )
    print(
        "Log directory: runs/test"
    )