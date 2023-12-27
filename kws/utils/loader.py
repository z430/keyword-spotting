from pathlib import Path
from typing import Dict

import yaml


def load_yaml(yaml_path: Path) -> Dict:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config