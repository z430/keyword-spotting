"""Model implementations for keyword spotting."""

import abc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kws.common.errors import ModelError


class KeywordSpottingModel(nn.Module, abc.ABC):
    """Base class for all keyword spotting models.

    This abstract class defines the interface that all KWS models should implement.
    """

    def __init__(self, num_classes: int, **kwargs):
        """Initialize the base model.

        Args:
            num_classes: Number of output classes
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            Output logits
        """
        pass

    def save(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save the model weights

        Raises:
            ModelError: If there's an error saving the model
        """
        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise ModelError(f"Failed to save model to {path}: {str(e)}") from e

    def load(self, path: str, device: Optional[torch.device] = None) -> None:
        """Load model weights from file.

        Args:
            path: Path to the model weights
            device: Device to load the model to

        Raises:
            ModelError: If there's an error loading the model
        """
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
        except Exception as e:
            raise ModelError(f"Failed to load model from {path}: {str(e)}") from e


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """Initialize the block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for depthwise convolution
            stride: Stride for depthwise convolution
            padding: Padding for both convolutions
        """
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(x)


class DepthwiseSeparableConv(KeywordSpottingModel):
    """Keyword spotting model using depthwise separable convolutions."""

    def __init__(self, num_classes: int = 12, input_channels: int = 1):
        """Initialize the model.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for mono audio)
        """
        super().__init__(num_classes)

        # First standard convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=10, stride=2, padding=2)

        # Depthwise separable convolution blocks
        self.dsconv1 = DepthwiseSeparableConvBlock(64, 64)
        self.dsconv2 = DepthwiseSeparableConvBlock(64, 64)
        self.dsconv3 = DepthwiseSeparableConvBlock(64, 64)
        self.dsconv4 = DepthwiseSeparableConvBlock(64, 64)

        # Classification layer
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 1, time, features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # First convolution
        x = F.relu(self.conv1(x))

        # Depthwise separable convolutions
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)

        # Global pooling and classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
