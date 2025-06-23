"""Training utilities for keyword spotting models."""

import abc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from kws.common.errors import handle_error, KWSError
from kws.libs.models import KeywordSpottingModel


class TrainingError(KWSError):
    """Error during model training."""

    def __init__(self, message: str = "Training error"):
        super().__init__(f"Training error: {message}")


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Basic training parameters
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

    # Optimization parameters
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "adam", "sgd", etc.
    scheduler: Optional[str] = None  # "cosine", "step", etc.
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Logging and checkpoints
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    use_clearml: bool = False
    project_name: str = "KeywordSpotting"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": self.shuffle,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scheduler_params": self.scheduler_params,
            "checkpoint_dir": self.checkpoint_dir,
            "log_interval": self.log_interval,
            "use_clearml": self.use_clearml,
            "project_name": self.project_name,
        }


class Trainer(abc.ABC):
    """Base abstract trainer class for keyword spotting models."""

    def __init__(
        self,
        model: KeywordSpottingModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_acc": 0.0,
            "best_epoch": 0,
        }

        # Setup criterion, optimizer, and scheduler
        self._setup_training()

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _setup_training(self):
        """Set up criterion, optimizer, and scheduler."""
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Optimizer
        if self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Learning rate scheduler
        self.scheduler = None
        if self.config.scheduler:
            if self.config.scheduler.lower() == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs,
                    **self.config.scheduler_params,
                )
            elif self.config.scheduler.lower() == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, **self.config.scheduler_params
                )
            else:
                raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    @abc.abstractmethod
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy)
        """
        pass

    @abc.abstractmethod
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy)
        """
        pass

    def train(self) -> Dict[str, List[float]]:
        """Train the model for the specified number of epochs.

        Returns:
            Dictionary of metrics
        """
        try:
            logger.info(f"Training on device: {self.device}")
            logger.info(f"Training for {self.config.epochs} epochs")

            for epoch in range(self.config.epochs):
                # Train one epoch
                train_loss, train_acc = self.train_epoch(epoch)
                self.metrics["train_loss"].append(train_loss)
                self.metrics["train_acc"].append(train_acc)

                # Validate
                val_loss, val_acc = self.validate(epoch)
                self.metrics["val_loss"].append(val_loss)
                self.metrics["val_acc"].append(val_acc)

                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Save best model
                if val_acc > self.metrics["best_val_acc"]:
                    self.metrics["best_val_acc"] = val_acc
                    self.metrics["best_epoch"] = epoch
                    self.save_checkpoint(f"best_model.pth")

                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )

            # Save final model
            self.save_checkpoint("final_model.pth")

            logger.info(
                f"Training completed. Best validation accuracy: "
                f"{self.metrics['best_val_acc']:.2f}% at epoch {self.metrics['best_epoch'] + 1}"
            )

            return self.metrics

        except Exception as e:
            handle_error(e, TrainingError, "Error during training")
            raise

    def save_checkpoint(self, filename: str) -> str:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file

        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": self.metrics,
                "config": self.config.to_dict(),
                "epoch": self.metrics.get("best_epoch", 0),
            },
            checkpoint_path,
        )
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = checkpoint.get("metrics", self.metrics)


class KWSTrainer(Trainer):
    """Standard trainer for keyword spotting models."""

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(
            self.train_loader,
            unit="batch",
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
        ) as progress:
            for batch_idx, (inputs, labels) in enumerate(progress):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100 * correct / total
                progress.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.2f}%"})

                # Log metrics at intervals
                if (
                    self.config.use_clearml
                    and batch_idx % self.config.log_interval == 0
                ):
                    try:
                        from clearml import Logger

                        Logger.current_logger().report_scalar(
                            "train/loss",
                            "batch loss",
                            value=avg_loss,
                            iteration=batch_idx + epoch * len(self.train_loader),
                        )
                    except (ImportError, Exception) as e:
                        logger.warning(f"Failed to log to ClearML: {e}")

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total

        # Log epoch metrics
        if self.config.use_clearml:
            try:
                from clearml import Logger

                Logger.current_logger().report_scalar(
                    "train/loss", "epoch loss", value=epoch_loss, iteration=epoch
                )
                Logger.current_logger().report_scalar(
                    "train/accuracy", "epoch acc", value=epoch_acc, iteration=epoch
                )
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to log to ClearML: {e}")

        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Update statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate validation metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total

        # Log validation metrics
        if self.config.use_clearml:
            try:
                from clearml import Logger

                Logger.current_logger().report_scalar(
                    "val/loss", "Val Loss", value=val_loss, iteration=epoch
                )
                Logger.current_logger().report_scalar(
                    "val/accuracy", "Val Acc", value=val_acc, iteration=epoch
                )
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to log to ClearML: {e}")

        return val_loss, val_acc
