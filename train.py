import argparse
from pathlib import Path
from dataclasses import asdict

from loguru import logger
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from clearml import Task, Logger
from datetime import datetime

from kws.datasets.speech_commands import DatasetConfig, SpeechCommandDataset
from kws.libs.dataloader import SpeechCommandsLoader
from kws.libs.signal_handler import AudioConfig, AudioProcessor
from kws.libs.models import DepthwiseSeparableConv

device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = running_loss / (tepoch.format_dict["n"] + 1)
                    tepoch.set_postfix({"Loss": f"{loss:.4f}"})
                    Logger.current_logger().report_scalar(
                        f"train/loss",
                        "batch loss",
                        value=loss,
                        iteration=tepoch.format_dict["n"],
                    )

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            Logger.current_logger().report_scalar(
                f"train/loss", "epoch loss", value=epoch_loss, iteration=epoch
            )
            Logger.current_logger().report_scalar(
                f"train/accuracy", "epoch acc", value=epoch_accuracy, iteration=epoch
            )

            self.evaluate(epoch)

        # Save model
        torch.save(self.model.state_dict(), "model.pth")

    def evaluate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the validation set: {100 * correct / total} %"
        )
        Logger.current_logger().report_scalar(
            f"val/accuracy", "Val Acc", value=100 * correct / total, iteration=epoch
        )


def train(opts):
    config = DatasetConfig()
    dataset = SpeechCommandDataset(config, Path("data/"))

    audio_config = AudioConfig()
    audio_processor = AudioProcessor(dataset.root_dir, audio_config)

    train_loader = DataLoader(
        SpeechCommandsLoader(dataset, audio_processor, "training"),
        batch_size=1028,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SpeechCommandsLoader(dataset, audio_processor, "validation"),
        batch_size=1028,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d%H-%M")
    task = Task.init(
        project_name="KeywordSpotting",
        task_name=f"{timestamp}-KWS-Train-DepthwiseSeparableConv",
    )

    task.connect(asdict(config))

    sample = next(iter(train_loader))
    input_shape = sample[0].shape

    logger.info(f"Training with input shape: {input_shape}")

    model = DepthwiseSeparableConv(
        num_classes=len(dataset.words_list), input_channels=1
    )

    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(100)

    task.update_output_model("model.pth")


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to model weights"
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_opt()
    train(opts)
