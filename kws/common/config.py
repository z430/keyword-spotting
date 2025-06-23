"""Configuration management utilities."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger

from kws.common.errors import KWSError


class ConfigError(KWSError):
    """Raised when there are issues with configuration."""

    def __init__(self, message: str = "Configuration error"):
        super().__init__(f"Configuration error: {message}")


def load_yaml_config(yaml_path: Path) -> Dict:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Dict containing the configuration

    Raises:
        ConfigError: If the file cannot be loaded or parsed
    """
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except (yaml.YAMLError, OSError) as e:
        logger.error(f"Failed to load configuration from {yaml_path}: {e}")
        raise ConfigError(f"Failed to load configuration from {yaml_path}") from e


def save_yaml_config(config: Dict, yaml_path: Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        yaml_path: Path where to save the YAML configuration

    Raises:
        ConfigError: If the file cannot be written
    """
    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    except (yaml.YAMLError, OSError) as e:
        logger.error(f"Failed to save configuration to {yaml_path}: {e}")
        raise ConfigError(f"Failed to save configuration to {yaml_path}") from e


def dataclass_to_dict(config: Any) -> Dict:
    """Convert a dataclass instance to a dictionary.

    Args:
        config: The dataclass instance to convert

    Returns:
        Dict representation of the dataclass
    """
    return asdict(config)
